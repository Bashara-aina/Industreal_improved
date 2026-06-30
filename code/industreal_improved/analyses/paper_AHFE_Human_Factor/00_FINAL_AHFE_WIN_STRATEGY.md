# AHFE 2026 Hawaii — POPW Win Strategy: Complete Research Synthesis

## Generated 2026-06-28 from 8-agent deep research (4 Layer-1 + 4 Layer-2)

---

## CRITICAL FINDING #1: HALLUCINATED CITATION IN CURRENT DRAFT

Your paper cites **"Parker et al. CHI 2017"** on piecework history. This paper does not exist. The real paper is:

> Alkhatib, A., Bernstein, M.S. & Levi, M. (2017). "Examining Crowd Work and Gig Work Through The Historical Lens of Piecework." *CHI 2017*, pp. 4599-4616. DOI: 10.1145/3025453.3025974

Replace immediately. The authors, venue, year, and content are different.

---

## COMPETITOR COMPARISONS

### Which systems to compare against (verified real papers)

| System | Venue | What it does | GPU/specs | Multi-task? |
|--------|-------|-------------|-----------|-------------|
| **ViMAT** | ICIAP 2025 | YOLOv8 + probabilistic reasoning for assembly monitoring | No GPU spec | No (detection only) |
| **IFAS** | J Intell Manuf 2026 | Screw fastening supervision only | Low-cost cameras | No (single task) |
| **Li et al.** | Machines 2025 | Swin Transformer multi-task defect detection | **GTX 3060 6GB**, 28.6 FPS | Yes (anomaly + classify) |
| **Resilient Assembly** | Digital 2026 | YOLO11-nano synthetic-to-real | Jetson Orin Nano | No (detection only) |
| **POPW (you)** | AHFE 2026 | Detection + Pose + Activity + PSR + Head Pose | RTX 3060 12GB, 4.8 FPS | **Yes (5 tasks)** |

**Warning on ENIGMA-360**: Your paper cites it as a "competitor assembly verification system." It is a **dataset paper** (arXiv:2603.09741, CVPR 2026 EgoVis Workshop). Re-classify or remove — citing it as a competitor weakens credibility.

**Warning on "first consumer-GPU" claim**: Li et al. (Machines, 2025) published a multi-task system on GTX 3060 for industrial defect detection (99.21% accuracy, 28.6 FPS). Your claim must be narrower: "first single-model multi-task ASSEMBLY VERIFICATION system combining detection + pose + activity + PSR + head pose on a consumer GPU."

---

## PAPER COMPARISON BENCHMARKS — WHAT YOU MUST ADD

### Missing benchmarks (reviewers will flag)

| Priority | Benchmark | Why | What to report |
|----------|-----------|-----|----------------|
| **CRITICAL** | **MECCANO** action recognition | Standard industrial-like assembly benchmark | Top-1/Top-5 accuracy |
| **CRITICAL** | **IKEA ASM** | Most cited furniture assembly benchmark | Frame acc, macro-recall |
| **HIGH** | **Assembly101** | Largest assembly dataset (513h) | Action recognition, segmentation F1 |
| **HIGH** | **HA-ViD** | Real industrial GAB (NeurIPS 2023) | Action recognition, segmentation |
| **MEDIUM** | **IndEgo** | Largest industrial ego dataset (NeurIPS 2025) | Procedure understanding |
| **MEDIUM** | **IMPACT** | Real assembly with anomaly (ACM MM 2026) | PSR, anomaly detection |

### Missing SOTA baselines (reviewers will flag)

| Paper | Venue | Task | Why you must cite |
|-------|-------|------|-------------------|
| **STORM-PSR** | CVIU 2025 | PSR | Directly on IndustReal — current PSR SOTA: POS 0.812, F1 0.901, tau 15.5s |
| **EgoPack** | CVPR 2024 | Egocentric MTL | Closest work to yours — shared backbone for Ego4D tasks |
| **ConsMTL** | CVPR 2025 | Gradient conflict resolution | SOTA for MTL optimization — addresses your multi-task collapse |
| **UW-SO** | IJCV 2025 | Loss weighting | Fixes known Kendall UW failures (overfitting, rigid assumptions) |
| **CAGrad** | NeurIPS 2021 | Gradient manipulation | Standard gradient-based MTL baseline |
| **Nash-MTL** | ICML 2022 | Gradient bargaining | Standard gradient-based MTL baseline |
| **PCGrad** | NeurIPS 2020 | Gradient surgery | Foundational gradient manipulation work |
| **MTLoRA** | CVPR 2024 | Parameter-efficient MTL | 3.6x fewer params — challenges your approach |
| **PromptonomyViT** | WACV 2024 | Prompt-based MTL | Alternative architectural choice challenging your heads design |
| **InterroGate** | BMVC 2024 | Learnable gating | Automates your manually designed sharing pattern |
| **ASDF** | ISMAR 2024 | Assembly state detection | Shows pose-state co-dependency directly relevant to POPW |
| **IndEgo dataset** | NeurIPS 2025 | Industrial ego benchmark | Largest relevant dataset — must acknowledge |
| **IMPACT dataset** | arXiv 2026 | Assembly benchmark | Real industrial with anomaly — most recent |

---

## AHFE EIC TRACK — ETHICAL FRAMEWORK

### Track specific info

- **Track name**: Ethical Issues and Considerations in Human Factors (EIC)
- **Chair**: Andreas Wolkenstein, Institute of Ethics, LMU Munich
- **Reviewer expectations**: Substantive ethical theory, human-factors grounding, real tensions (not platitudes), interdisciplinary, practical implications
- **Your EIC hook**: The core contribution is the ethical framework (IEEE 7005-2021), not the technical system

### Papers you MUST cite for the ethics section

1. **Sebastian, Ehinger & Miller (2025)** — "Do we need watchful eyes on our workers? Ethics of CV for workplace surveillance." *AI and Ethics*. DOI: 10.1007/s43681-025-00726-4. The definitive ethics-of-CV-surveillance paper. Proposes intent- and priority-based ethical framework for CV workplace monitoring.

2. **Wolkenstein, A. (2024)** — "Healthy Mistrust: Medical Black Box Algorithms." *Cambridge Quarterly of Healthcare Ethics*, 33(3), 370-379. DOI: 10.1017/S0963180123000646. Your track chair's own paper. Argues algorithmic systems can serve as epistemic safeguards.

3. **Callari et al. (2024)** — "An ethical framework for human-robot collaboration for future people-centric manufacturing." *Technology in Society*, 79, 102680. Three micro-level principles: autonomy, authority, agency.

4. **Li & Wang (2026)** — "The ethics of algorithmic visibility: power and resistance in AI surveillance at construction sites." *AI and Ethics*, 6, 109. Foucaultian analysis of digital panopticism in workplace monitoring.

5. **Floridi, L. (2013)** — "Distributed morality in an information society." *Science and Engineering Ethics*, 19(3), 727-743. Directly applicable to multi-agent blockchain-AI monitoring systems.

6. **Floridi, L. (2016)** — "Faultless responsibility." *Philosophical Transactions of the Royal Society A*, 374, 20160112. Allocating moral responsibility in distributed systems.

7. **Milanez, Lemmens & Ruggiu (2025)** — "Algorithmic management in the workplace." *OECD AI Papers*, No. 31. DOI: 10.1787/287c13c4-en. Survey of 6,000+ firms — empirical grounding.

8. **Brintrup et al. (2025)** — "Trustworthy, responsible and ethical AI in manufacturing." *Data-Centric Engineering*, Cambridge. Systematic mapping with worker monitoring failure examples.

9. **De Coninck et al. (2026)** — "Privacy-Preserving Computer Vision for Industry." *AAAI 2026*. Validated privacy framework for industrial CV.

10. **Bathaeijavareshk et al. (2025)** — "Multidisciplinary Perspectives on Ethical AI-Enabled HRI in Manufacturing." *AHFE 2025 proceedings*. DOI: 10.54941/ahfe1006381. Published in AHFE — shows venue precedent.

### Blockchain-specific ethics papers

11. **Sharif, M.M. & Ghodoosi, F. (2022)** — "The Ethics of Blockchain in Organizations." *Journal of Business Ethics*, 178(4). DOI: 10.1007/s10551-022-05058-5. Examines blockchain in HR through virtue ethics, utilitarianism, deontology, contractarianism.

12. **Merk et al. (2026)** — "A workers' inquiry in DAOs." *Internet Policy Review*, 15(1). Empirical study: DAO work is unstable, undercompensated, unpredictable.

13. **Hawashin et al. (2025)** — "ML + blockchain for trusted detection of excessive working hours." *Technology in Society*, 81, 102959. 96.6% accuracy — directly relevant to your approach.

### Ethical gaps in current draft

| Missing argument | Why critical |
|----------------|-------------|
| **Digital panopticism** (Foucault/Li & Wang) | Blockchain immutability could make panopticism WORSE |
| **Distributed morality** (Floridi) | Multi-agent blockchain-CV system has no single responsible agent |
| **Worker autonomy** (Callari framework) | Monitoring inherently reduces autonomy — what margins of freedom remain? |
| **Blockchain vs GDPR** (right to erasure) | Immutability conflicts with GDPR Article 17 — address this tension |
| **Monitoring vs surveillance distinction** | Cite Jidoka Tech / Protex AI criteria |
| **Optifye.ai precedent** | YC startup deleted demo after "sweatshop software" accusations (Feb 2025) |
| **Gender/demographic bias** | Brintrup et al.: body-movement CV performs poorly on female workers |
| **Union/collective bargaining** | Is the blockchain transparent to worker representatives? |

---

## CONSUMER GPU POSITIONING

### Real RTX 3060 benchmarks from published papers

| Paper | Model | GPU | FPS | Resolution |
|-------|-------|-----|-----|-----------|
| Nature Scientific Reports 2026 | YOLOv8 | RTX 3060 | 83.3 | 640x640 |
| YOLOv8 TensorRT benchmarks | YOLOv8m FP16 | RTX 3060 | 30.0 | 640x640 |
| YOLOv8 TensorRT benchmarks | YOLOv8x FP16 | RTX 3060 | 15.0 | 640x640 |
| SpecPicks 2026 | YOLOv11x | RTX 3060 | 184 | 640x640 |
| Li et al. 2025 (Machines) | Swin Transformer | GTX 3060 | 28.6 | 248x248 |
| **POPW (yours)** | ConvNeXt-T + 5 heads | RTX 3060 | **4.8** | 720x1280 |

Your FPS is low because of 720x1280 resolution and 5 task heads. Acknowledge this and frame it as the cost of multi-task coverage.

---

## SUMMARY: KEY ACTIONS TO WIN AT AHFE

1. **Fix hallucinated citation**: Replace "Parker et al. CHI 2017" with Alkhatib, Bernstein & Levi (2017)
2. **Add 14+ missing citations** (especially EgoPack, STORM-PSR, ConsMTL, UW-SO)
3. **Reclassify ENIGMA-360** as dataset, not competitor
4. **Narrow "first consumer-GPU" claim** to "first single-model multi-task ASSEMBLY VERIFICATION on consumer GPU"
5. **Add MECCANO + IKEA ASM + Assembly101 comparisons**
6. **Add STORM-PSR comparison** for PSR metrics
7. **Strengthen ethical section**: add Floridi distributed morality, blockchain-GDPR tension, panopticism critique, worker agency
8. **Cite EIC track chair** (Wolkenstein 2024) and Sebastian et al. (2025)
9. **Report RTX 3060 benchmarks** (FPS, VRAM, comparison vs sequential single-task models)
10. **Address Kendall UW limitations** and compare against UW-SO

---

## PRIORITY ORDER FOR FIXES

| Priority | Fix | Time | Impact |
|----------|-----|------|--------|
| P0 | Fix hallucinated "Parker" citation | 5 min | Critical — caught citation could get rejected |
| P0 | Add STORM-PSR comparison | 1-2 days | Most critical missing technical comparison |
| P1 | Add missing MTL citations (EgoPack, ConsMTL, UW-SO, CAGrad, etc.) | 1-2 hours | Reviewers will check related work |
| P1 | Reclassify ENIGMA-360 | 5 min | Category error damages credibility |
| P1 | Add Wolkenstein 2024 + Sebastian 2025 to ethics section | 30 min | Shows track awareness |
| P1 | Add Floridi distributed morality + blockchain-GDPR tension | 1 hour | Core ethical argument missing |
| P2 | Add MECCANO + IKEA ASM benchmarks | 1-3 days | Broadens evaluation significantly |
| P2 | Report RTX 3060 inference benchmarks | 1 day | Supports consumer GPU claim |
| P2 | Add Li et al. 2025 as competitor | 30 min | Acknowledges existing consumer GPU MTL work |
| P2 | Narrow "first" claim | 15 min | Avoids reviewer pushback |
| P3 | Add worker agency / panopticism / bias arguments | 1 hour | Strengthens EIC framing |
| P3 | Add IndEgo + IMPACT dataset references | 15 min | Shows awareness of latest benchmarks |
