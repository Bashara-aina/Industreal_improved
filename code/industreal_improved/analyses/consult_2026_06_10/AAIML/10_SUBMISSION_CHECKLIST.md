# AAIML 2027 -- Submission Checklist

**Paper**: POPW: A Multi-Task Deep Learning Framework for Assembly Verification
**Deadline**: October 10, 2026 (102 days from June 30)
**Format**: IEEE 2-column, 6-10 pages (max 6 free, up to 4 extra at $70/page)
**Submission**: EasyChair (https://aaiml.net/submission)

---

## Format Compliance

### IEEE Format
- [ ] Paper is in IEEE 2-column format (IEEEtran.cls, conference option)
- [ ] Font size: 10pt (no smaller)
- [ ] Title centered, initial caps, no bold/italic formatting
- [ ] Author names and affiliation match registered EasyChair account
- [ ] No page numbers, headers, or footers (IEEE removes them)
- [ ] Figures are embedded as vector PDF or 300+ DPI PNG
- [ ] Figure captions are centered, below figures (IEEE convention)
- [ ] Table captions are above tables (IEEE convention)
- [ ] Bibliography uses IEEE style (numbers in brackets, [1], [2])
- [ ] DOI/URL formatting in bibliography follows IEEE guidelines
- [ ] Paper length: 6-10 pages (aim for 8)
- [ ] Abstract: 150-200 words, no citations, no math symbols
- [ ] Keywords: 4-6 terms, comma-separated

### PDF Generation
- [ ] PDF generated from LaTeX (not Word)
- [ ] PDF embeds all fonts (IEEE requirement)
- [ ] All hyperlinks work (URLs, citations, cross-references)
- [ ] PDF file size under 10 MB (IEEE Xplore limit)
- [ ] No special characters in filename: popw_aaiml2027.pdf
- [ ] PDF title metadata matches paper title

---

## Content Completeness

### Required Sections
- [ ] Abstract (150-200 words)
- [ ] Keywords (4-6, IEEE-compliant)
- [ ] Introduction (Section 1): problem, approach, contributions
- [ ] Related Work (Section 2): MTL, assembly, blockchain, competitors table
- [ ] System Architecture (Section 3): backbone, 5 heads, FiLM, training
- [ ] Experiments (Section 4): primary results, SOTA comparison, ablations
- [ ] Blockchain (Section 5): x402 pipeline, latency, cost
- [ ] Factory Pilot (Section 6): study design, quantitative, qualitative
- [ ] Ethical Framework (Section 7): IEEE 7005 mapping, failure modes
- [ ] Discussion (Section 8): limitations, tradeoffs, future work
- [ ] Conclusion (Section 9): summary, impact
- [ ] References (30+, IEEE format)
- [ ] Data and Code Availability statement

### Required Data in Results Section
- [ ] Detection present-class mAP50 with bootstrap 95% CI
- [ ] Detection standard mAP50
- [ ] Head pose angular error with standard deviation
- [ ] Activity Top-1 and Top-5 accuracy
- [ ] Parameter count (53M trainable, 76M total)
- [ ] GFLOPs per frame (93)
- [ ] FPS on RTX 3060 (4.8)
- [ ] Ablation A: single-task vs multi-task with delta
- [ ] Ablation B: with/without FiLM with p-value
- [ ] Blockchain end-to-end latency (537ms) with N and sigma
- [ ] Blockchain gas cost range ($0.0002-$0.001)
- [ ] Pilot: opt-out rate, SUS, NASA-TLX pre/post, Trust, Surveillance perception
- [ ] Three-seed variance (PROMISED for camera-ready)

---

## Figures and Tables

### Figures Checklist
- [ ] Figure 1: System architecture diagram (PDF vector)
- [ ] Figure 2: Detection confusion matrix (24x24 heatmap)
- [ ] Figure 3: Ablation A comparison (bar chart)
- [ ] Figure 4: Ablation B FiLM effect (bar chart)
- [ ] Figure 5: Cost comparison (horizontal bar chart)
- [ ] Figure 6: Pilot dashboard (multi-panel)
- [ ] Figure 7: Training curves (optional, camera-ready)
- [ ] Figure 8: Blockchain pipeline diagram (optional, camera-ready)

### Tables Checklist
- [ ] Table 1: Competitor analysis (Section 2)
- [ ] Table 2: Staged training protocol (Section 3)
- [ ] Table 3: Primary results (Section 4)
- [ ] Table 4: SOTA comparison (Section 4)
- [ ] Table 5: Ablation results (Section 4)
- [ ] Table 6: Pilot results (Section 6)
- [ ] Table 7: IEEE 7005 mapping (Section 7)

### Figure Quality
- [ ] All figures are vector PDF (preferred) or 300+ DPI PNG
- [ ] Font sizes in figures match or exceed IEEE minimum
- [ ] Color figures use colorblind-friendly palettes
- [ ] All figures have descriptive captions
- [ ] Figure placement uses `[htbp]` to avoid orphan pages
- [ ] No figure exceeds column width (3.5 inches for single, 7.5 inches for double)

---

## Code and Data Release

### GitHub Repository (https://github.com/bashara-aina/popw)
- [ ] Model weights uploaded (Hugging Face or GitHub Releases)
- [ ] Inference script with example
- [ ] Training script with configuration
- [ ] Evaluation script reproducing all reported metrics
- [ ] README with setup instructions
- [ ] Requirements file (pip or conda)
- [ ] Dockerfile or environment specification
- [ ] License file (MIT, Apache 2.0, or similar)
- [ ] README includes citation info for the paper
- [ ] No hardcoded paths in scripts
- [ ] x402 Solana pipeline code separate and documented

### Reproducibility
- [ ] Data splits (70/15/15) documented and reproducible
- [ ] Random seed(s) specified
- [ ] Training hyperparameters in config file
- [ ] Evaluation command exactly reproduces Table 3
- [ ] Evaluation command has `--seed` argument for multi-seed runs

---

## Ethical Compliance

### Human Subjects
- [ ] IRB approval documented (or exemption noted)
- [ ] Informed consent procedure described
- [ ] Opt-out mechanism specified
- [ ] Worker demographics reported (age range, gender, experience)
- [ ] Compensation for participation disclosed (if any)
- [ ] Privacy protections: edge-only processing, no face data

### IEEE 7005-2021 Compliance
- [ ] All 5 IEEE 7005 principles listed with implementation status
- [ ] (P) items clearly marked as planned, not promised
- [ ] Data governance: edge-only processing stated
- [ ] Accountability: blockchain immutability stated

### General AI Ethics
- [ ] Surveillance concern addressed
- [ ] Worker autonomy and consent respected
- [ ] Failure modes discussed (function creep, panoptic pressure, etc.)
- [ ] Citation to relevant ethics literature (Sebastian, Milanez, Floridi)

---

## Reviewer Defense Readiness

### Top 10 Anticipated Reviewer Questions
- [ ] Q1: "Why not just use a database instead of blockchain?" -- answered in Section 5
- [ ] Q2: "Detection is too low to be useful" -- answered in Section 4
- [ ] Q3: "Is this really multi-task or just five separate heads?" -- answered in Section 3
- [ ] Q4: "Why not use a larger backbone?" -- answered in Section 3 (consumer GPU constraint)
- [ ] Q5: "How does this compare to [Random 2026 paper]?" -- check arXiv weekly for new submissions
- [ ] Q6: "What about lighting variation, occlusion, etc.?" -- addressed in Discussion
- [ ] Q7: "Can workers game the system?" -- addressed in Section 5 (blockchain prevents)
- [ ] Q8: "4.8 FPS is too slow" -- answered in Section 3 and Discussion
- [ ] Q9: "20 workers is not generalizable" -- answered in Section 6 and Discussion
- [ ] Q10: "What about [relevant paper not cited]?" -- check citation network in 05_CITATION_NETWORK.md

---

## Final Review (One Week Before Submission)

### Read-Through Checklist
- [ ] No placeholder text ("TODO," "FIXME," "TBD")
- [ ] No empty sections or subsections
- [ ] All table cells filled (no empty cells)
- [ ] All cross-references correct (Section \ref{} matches actual section)
- [ ] All bibliographic references cited in text
- [ ] No duplicate citations (same paper cited twice with different bibtex keys)
- [ ] Equation numbering sequential
- [ ] All acronyms defined on first use
- [ ] Consistent terminology throughout (POPW, not PoPW except in citations)
- [ ] No first-person plural for single author (use "we" is fine in IEEE)
- [ ] No industry jargon unexplained (x402, DePIN, etc.)

### Technical Validation
- [ ] All reported numbers match evaluation output (re-run evaluation before submission)
- [ ] Three-seed results available (promised for camera-ready; state "preliminary single-seed")
- [ ] Bootstrap confidence intervals regenerated with latest data
- [ ] Figure scripts re-run with final data
- [ ] No arithmetic errors in tables (sums match totals, percentages add up)

### EasyChair Submission
- [ ] Author registered on EasyChair
- [ ] Paper uploaded as PDF
- [ ] All authors listed correctly
- [ ] Abstract copied (not regenerated from LaTeX)
- [ ] Keywords selected from AAIML menu
- [ ] Topics match paper content (select 3-5 relevant topics)
- [ ] Submission confirmation email received
- [ ] PDF visible in EasyChair preview (verify formatting)

---

## Post-Submission (Before Camera-Ready, Nov 30)

- [ ] Run three seeds for all experiments (highest priority)
- [ ] Generate three-seed variance for all metrics
- [ ] Add precision-recall curves with operating point analysis
- [ ] Add activity temporal smoothing results
- [ ] Add blockchain vs signed-log comparison
- [ ] Prepare camera-ready template with all figures
- [ ] Verify IEEE Xplore compliance (embed fonts, PDF/A if required)
- [ ] Pre-pay extra page fees if over 6 pages ($70/page, up to 4 extra)

---

## Daily Countdown

| Days to Deadline | Milestone | Status |
|-----------------|-----------|--------|
| 102 (Jun 30) | Strategy finalized | Done |
| 90 (Jul 12) | Architecture sections drafted | |
| 75 (Jul 27) | Pilot complete | |
| 60 (Aug 11) | Full draft v1 | |
| 45 (Aug 26) | All figures generated | |
| 30 (Sep 10) | Full draft v2 with all data | |
| 15 (Sep 25) | Final formatting | |
| 7 (Oct 3) | Final review | |
| 1 (Oct 9) | Last upload test | |
| **0 (Oct 10)** | **SUBMIT** | |
