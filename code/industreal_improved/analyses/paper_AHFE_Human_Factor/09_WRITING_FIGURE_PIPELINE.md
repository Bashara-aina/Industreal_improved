# Plan 9: Paper Writing and Figure Generation Pipeline

> **Deadline:** July 24, 2026 (camera-ready)
> **Target completion for draft:** July 15, 2026 (9 days buffer for revisions/formatting)
> **Format:** MS Word .docx via AHFE template, 10 pages maximum, 500-word abstract

---

## 1. Paper Section Writing Order (Dependency-Aware)

| Section | Pages | GPU Needed? | Can Start | Est. Hours | Priority |
|---------|-------|-------------|-----------|------------|----------|
| **Abstract** | 0.5 | No | **IMMEDIATELY** | 1 | Critical |
| **Section 6: Ethical Analysis** | 2 | No | **IMMEDIATELY** | 4 | **Highest** |
| **Section 1: Introduction** | 1 | No | **IMMEDIATELY** | 2 | High |
| **Section 2: Related Work** | 1.5 | No | **IMMEDIATELY** | 3 | High |
| **Section 7: Discussion/Limitations** | 1 | No | **IMMEDIATELY** | 2 | High |
| **Section 3: System Design** | 1.5 | No | **IMMEDIATELY** | 3 | Medium |
| **Section 5: Blockchain** | 1.5 | After devnet test | Jul 5 | 3 | Medium |
| **Section 4: Results** | 2 | After training | Jul 5-10 | 4 | Dependent |
| **References** | 0.5 | No | Ongoing | 2 | On-going |
| **Figures** | — | After training | Jul 10-15 | 8-12 | Dependent |
| **Formatting (docx)** | — | No | Jul 18-20 | 4 | Late |

### Writing Timeline

```
Jun 27-30: Write Sections 6, 1, 2, 7 (no GPU needed — these are the HF/ethics core)
           4 sections = ~5.5 pages of the 10 page paper = 55% done before any GPU results

Jul 1-5:   Write Section 3, draft Section 5 skeleton
           Fill Tables 1 (competitors), 4 (TCO), 6 (ethics) — no training data needed

Jul 5-10:  Training completes → Fill Section 4 with real numbers
           Fill Tables 2 (results), 3 (ablation), 5 (blockchain)

Jul 10-15: Generate Figures 1-6
           Complete references (45+ citations)
           FULL DRAFT COMPLETE

Jul 15-20: Sensei review → revisions
           LaTeX to Word conversion → AHFE template
           Consent form signature
```

---

## 2. Section-by-Section Writing Guide

### ABSTRACT (500 words, 1 paragraph)

**Rules:**
- Single paragraph, no line breaks
- NO abbreviations or CV jargon (replace "mAP" with "detection accuracy")
- First sentence states the human problem (fair compensation)
- Middle sentences state the technical contribution in accessible language
- Last sentence states the broader impact
- Keywords: 5-7, separated by semicolons

**Template from Plan 1 Section 1.2 — ready to write immediately.**

### SECTION 1: INTRODUCTION (1 page, ~500 words)

**Structure:**
1. Opening paragraph: The human problem (fair compensation, $10K barrier, fragmented systems)
2. Three gaps: Economic, Technical, Human
3. Our answer: POPW — one model, $299 GPU, five tasks, blockchain payments, ethics framework
4. Four enumerated contributions

**Writing style notes:**
- "Worker support" NOT "surveillance"
- "Fair compensation" NOT "automated payment"
- "Gaze monitoring" NOT "head pose tracking"
- No bullets — write in prose paragraphs

### SECTION 6: ETHICAL ANALYSIS (2 pages, ~1000 words) ★ HIGHEST PRIORITY

**This section is the reason the paper is in the EIC track. Write it first.**

**Subsection structure:**
- **6.1 Surveillance vs Empowerment** (0.5 page)
- **6.2 IEEE 7005-2021 Compliance Framework with Table 6** (0.5 page)
- **6.3 Algorithmic Fairness** (0.5 page)
- **6.4 Regulatory Alignment (EU AI Act, Platform Work Directive)** (0.25 page)
- **6.5 Economic Justice** (0.25 page)

**Key citations to include:**
- IEEE 7005-2021 (cite specific sections per Table 6)
- EU AI Act (high-risk classification for workplace AI)
- EU Platform Work Directive (automated monitoring transparency, transposed Dec 2026)
- Floridi & Cowls (2019) — 5 principles for AI in society
- The EIC track chair's own work: Wolkenstein, A. "Healthy Mistrust: Medical Black Box Algorithms" (2024)

### SECTION 4: RESULTS (2 pages, ~1000 words)

**Write this LAST — it depends on training numbers.**

**Table 2 preparation:**
```python
# After evaluation completes, extract these values:
metrics = {
    "det_mAP50_pc": "[from RF2 val]",
    "forward_angular_MAE_deg": "[from RF2 val]",
    "act_accuracy": "[from RF3 val]",
    "eff_params_m": "[from Phase 0]",
    "eff_gflops": "[from Phase 0]",
    "eff_fps": "[from Phase 0]",
}
# Fill into Table 2 template
```

---

## 3. Figure Generation Timeline

| Figure | Description | Tool | Est. Time | Can Start | Deadline |
|--------|-------------|------|-----------|-----------|----------|
| **Fig 1**: Application Scenario | Worker + camera + GPU + callouts | draw.io / Illustrator | 4-6h | **IMMEDIATELY** | Jul 10 |
| **Fig 2**: Payment Pipeline | CCTV → POPW → x402 → Solana → Wallet | draw.io | 1h | **IMMEDIATELY** | Jul 10 |
| **Fig 3**: Confusion Matrix | 24x24 heatmap | matplotlib | 1h | After training | Jul 12 |
| **Fig 4**: Cost Comparison | Bar chart | matplotlib | 1h | **IMMEDIATELY** (known prices) | Jul 10 |
| **Fig 5**: Head Pose Overlay | 4 frames with gaze arrows | Python + ffmpeg | 2-3h | After training | Jul 12 |
| **Fig 6**: Ethics Framework | 4-column diagram (draw.io) | draw.io | 1h | **IMMEDIATELY** | Jul 10 |

### Figure 1 Detailed Specification (Most Important Figure)

**Content (from left to right):**
1. Worker at workstation wearing egocentric camera on hard hat
2. Assembly components on workbench (beverage bottles, caps, labels)
3. Small PC labeled "RTX 3060 — $299 — Local Processing" with green checkmark
4. Callout 1: "Step 5/12: Attaching Cap — VERIFIED ✓" 
5. Callout 2: Arrow from camera showing "Gaze: Assembly Area"
6. Callout 3: Bounding boxes on components with labels
7. Dashboard inset (bottom right): "Today: 157 assemblies — $15.70 earned"

**Style:** Clean vector illustration, warm colors, human-centered (worker is front and center, not the hardware)

**Resolution:** 600 DPI, embedded in Word document

---

## 4. Table Population Timeline

| Table | Content | When to Fill | Source |
|-------|---------|-------------|--------|
| **Table 1**: Competitor Analysis | 6 approaches vs POPW | IMMEDIATELY | Known from literature |
| **Table 2**: Primary Results | 6 metrics + HF meaning | After training by Jul 10 | evaluate.py output |
| **Table 3**: Ablation A | Single vs multi-task | After ablation by Jul 5 | Ablation A eval |
| **Table 4**: 3-Year TCO | Cost comparison | IMMEDIATELY | Current market prices |
| **Table 5**: Payment Latency | 5 pipeline stages | After devnet test by Jul 5 | x402 measurement |
| **Table 6**: Ethical Principles | 6 + IEEE 7005 refs | IMMEDIATELY | IEEE 7005 standard |

---

## 5. References — Complete (45+ Papers)

**Ready to write immediately** with the following verified sources:

**Assembly Datasets (9):** IndustReal, STORM-PSR, IKEA ASM, Assembly101, MECCANO, HA-ViD, IndEgo, IMPACT, ENIGMA-360, OpenMarcie

**Multi-Task Video (3):** EgoPack, Differentiable Task Graph Learning, EgoIndAssembly

**Industry CV Systems (6):** ViMAT, IFAS, Resilient Assembly Supervision, Privacy-preserving CV for Industry, DELEGACT, CoViLLM

**AHFE Proceedings (7):** 
1. Papoutsakis et al. "Posture deviations in assembly" (AHFE 2024)
2. Luque et al. "AI-enhanced Ergonomics" (AHFE 2024)
3. Omri et al. "CV for Sustainable Manufacturing" (AHFE 2024)
4. Pontes et al. "Ergonomic posture tracking" (AHFE 2025)
5. "Assistive Assembly" (AHFE 2024) — Enhancing Worker Efficiency
6. "Evaluation of Feedback in Manual Assembly" (AHFE 2024)
7. "Multidisciplinary Perspectives on Ethical AI-Enabled HRI" (AHFE 2025)

**Blockchain + Manufacturing (5):** Konnex PoPW, PopChain, Materialize, Blockchain-Embedded SLA, DePIN tokenomics, SmartQC

**Ethics (7):** IEEE 7005-2021, Wolkenstein "Healthy Mistrust" (2024), Floridi & Cowls (2019), Parker "Piecework" (CHI 2017), Milanez "Algorithmic Management" (OECD 2025), EU AI Act, EU Platform Work Directive

---

## 6. Daily Word Count Target

| Date | Target Cumulative Words | Section |
|------|----------------------|---------|
| Jun 27 | 500 | Abstract |
| Jun 28 | 1,500 | Section 6 (Ethics) |
| Jun 29 | 2,500 | Section 1 (Intro) |
| Jun 30 | 4,000 | Section 2 (Related Work) |
| Jul 1 | 5,000 | Section 7 (Discussion) |
| Jul 2 | 5,500 | Section 3 architecture |
| Jul 5 | 6,500 | Section 5 (Blockchain) |
| Jul 10 | 8,000 | Section 4 (Results) + fill tables |
| Jul 12 | 8,500 | Figures done, references done |
| Jul 15 | 9,000 | **FULL DRAFT COMPLETE** |
| Jul 20 | 9,000 | Formatted, reviewed, ready |

**Target is ~9,000 words for 10 pages** (AHFE template fits ~900 words/page including figures/tables).
