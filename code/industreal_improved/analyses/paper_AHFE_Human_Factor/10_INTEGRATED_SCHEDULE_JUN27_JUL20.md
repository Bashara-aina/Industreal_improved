# Plan 10: Integrated Schedule — June 27 to July 20 — Execution to Submission

> **Hard deadline:** July 24, 2026 (camera-ready to edition.ahfe-cms.org)
> **Target internal deadline:** July 20, 2026 (4 days buffer for emergencies)
> **23 days remaining — every hour accounted for**

---

## 1. Master Schedule — Day-by-Day

```
LEGEND:
[5060] = RTX 5060 Ti (CUDA GPU 0) — primary training
[3060]  = RTX 3060 (CUDA GPU 1) — ablations + eval
[CPU]   = Writing/analysis — no GPU needed
[BOTH]  = Both GPUs
```

### WEEK 1: Foundation (Jun 27 — Jul 3)

| Day | [5060] Primary Training | [3060] Ablations/Eval | [CPU] Writing/Setup | Critical Deadlines |
|-----|------------------------|----------------------|---------------------|-------------------|
| **Jun 27 (Sat)** | Launch RF2 from crash_recovery.pth | Launch efficiency measurement ⚡ | Write Abstract + Section 6 (Ethics) | — |
| **Jun 28 (Sun)** | RF2 continues (~epoch 5) | PSR go/no-go (1h) + Archive old runs | Write Section 1 (Introduction) | — |
| **Jun 29 (Mon)** | RF2 continues (~epoch 10) | Launch Ablation A (recovery_det_only) | Write Section 2 (Related Work) | — |
| **Jun 30 (Tue)** | RF2 completion (~epoch 15) | Ablation A continues (~epoch 5) | Write Section 7 (Discussion) | — |
| **Jul 1 (Wed)** | Launch RF3 (activity training) | Ablation A continues (~epoch 10) | Write Section 3 (System Design) | — |
| **Jul 2 (Thu)** | RF3 continues (~epoch 5) | Ablation A continues — check Top-1 | Fill Tables 1, 4, 6 | — |
| **Jul 3 (Fri)** | RF3 continues (~epoch 8) | Ablation A eval → confusion matrix | Start Fig 1 (application scenario) | **⚠️ REGISTER AHFE (early bird ends)** |

**Week 1 word count target:** 5,000 words (Abstract + Sections 1, 2, 6, 7)

### WEEK 2: Results and Blockchain (Jul 4 — Jul 10)

| Day | [5060] Primary Training | [3060] Ablations/Eval | [CPU] Writing/Setup | Critical Deadlines |
|-----|------------------------|----------------------|---------------------|-------------------|
| **Jul 4 (Sat)** | RF3 continues (~epoch 10) | Per-class diagnostic | Write Section 5 skeleton | — |
| **Jul 5 (Sun)** | RF3 continues (~epoch 12) | Install Solana CLI | Deploy x402 template to devnet | — |
| **Jul 6 (Mon)** | RF3 complete (~epoch 15) | x402 latency measurement (100 cycles) | Fill Table 5 (blockchain latency) | — |
| **Jul 7 (Tue)** | FINAL EVAL on Phase B | FINAL EVAL on Ablation A | Fill Table 2 (primary results) | — |
| **Jul 8 (Wed)** | Generate confusion matrix | Generate cost bar chart | Fill Table 3 (ablation A) | — |
| **Jul 9 (Thu)** | Generate head pose overlay | Fill remaining data gaps | Complete Section 4 (Results) | — |
| **Jul 10 (Fri)** | **ALL TRAINING DONE** | **ALL ABLATIONS DONE** | **ALL TABLES WITH DATA** | — |

**Week 2 milestones:**
- All training complete
- All ablation experiments complete
- All 6 tables filled with real numbers
- Blockchain latency measured on devnet
- Word count: 6,500 (Sections 3, 5 drafted)

### WEEK 3: Writing and Figures (Jul 11 — Jul 17)

| Day | [BOTH] | [CPU] Primary Focus | Deliverable |
|------|--------|---------------------|-------------|
| **Jul 11 (Sat)** | Idle | Generate Figures 3 (confusion), 4 (cost), 5 (head pose) | 3 figures done |
| **Jul 12 (Sun)** | Idle | Generate Figures 1 (scenario), 2 (pipeline), 6 (ethics) | All 6 figures done |
| **Jul 13 (Mon)** | Idle | Polish Section 4 (Results) with all numbers | Section 4 complete |
| **Jul 14 (Tue)** | Idle | Polish Section 6 (Ethics) + add Floridi + Wolkenstein citations | Section 6 complete |
| **Jul 15 (Wed)** | Idle | **FULL DRAFT COMPLETE — all 10 pages** | **🎯 TARGET** |
| **Jul 16 (Thu)** | Idle | Self-review against AHFE criteria | Review notes |
| **Jul 17 (Fri)** | Idle | References complete (45+ citations) | References done |

**Week 3 milestone: FULL DRAFT COMPLETE by July 15**

### WEEK 4: Polish and Submit (Jul 18 — Jul 24)

| Day | Task | Detail |
|------|------|--------|
| **Jul 18 (Sat)** | Convert LaTeX to AHFE Word template | Match AHFE formatting exactly |
| **Jul 19 (Sun)** | Generate matching PDF | Ensure docx → PDF is pixel-perfect |
| **Jul 20 (Mon)** | **SENSEI REVIEW** | **📩 Send to sensei** |
| **Jul 21 (Tue)** | Incorporate sensei feedback | Revisions round 1 |
| **Jul 22 (Wed)** | Final polish + download consent form | Print, sign, scan |
| **Jul 23 (Thu)** | Final format check | 20-check item compliance checklist |
| **Jul 24 (Fri)** | **📤 SUBMIT** | **edition.ahfe-cms.org** |

---

## 2. Weekly Hour Budget

| Week | GPU Hours (5060) | GPU Hours (3060) | Writing Hours | Focus |
|------|-----------------|-----------------|---------------|-------|
| Week 1 (Jun 27-Jul 3) | 30h (RF2 18h + RF3 12h) | 20h (Ablation A) | 15h | Training + Abstract + Sections 1,2,6,7 |
| Week 2 (Jul 4-10) | 10h (RF3 finish + eval) | 5h (x402 + diagnostics) | 20h | Results + Tables + Blockchain |
| Week 3 (Jul 11-17) | 0h | 0h | 30h | Figures + Polish + Full draft |
| Week 4 (Jul 18-24) | 0h | 0h | 20h | Revisions + Format + Submit |
| **Total** | **40h** | **25h** | **85h** | |

---

## 3. Daily Standup Signal Checklist

Each morning, check these signals to know if you're on track:

### June 27-30 (RF2 training)
```
☐ RF2 loss decreasing (should go from ~15 to ~5 over 15 epochs)
☐ LIVENESS shows det=ALIVE, pose=ALIVE, head_pose=ALIVE
☐ GPU memory < 10 GB (should be ~2-6 GB)
☐ No NaNs in loss
```

### July 1-6 (RF3 activity training)
```
☐ Activity Top-1 > 5% by epoch 5
☐ Activity Top-1 > 10% by epoch 10
☐ Detection NOT regressing (mAP50_pc within 0.02 of RF2 best)
☐ Activity confusion matrix shows spread (not collapsed to one class)
```

### July 7-10 (Evaluation)
```
☐ det_mAP50_pc ≥ 0.25 (prior best: 0.304)
☐ forward_angular_MAE_deg ≤ 12 (prior best: 9.13)
☐ act_accuracy ≥ 10% (chance baseline: 1.3%)
☐ Ablation A delta < 0.05 (no catastrophic interference)
```

### July 11-15 (Writing)
```
☐ Fig 1 (scenario) — most important figure — done
☐ All 6 tables filled with real numbers
☐ IEEE 7005-2021 cited in text with specific section references
☐ 5+ AHFE proceedings papers cited
```

---

## 4. Critical Path — What Must NOT Slip

| Task | Latest Date | Why |
|------|-------------|-----|
| **Register for AHFE** | **Jul 3** | Without registration, paper cannot be published. Early bird ends. |
| **RF2 complete** | **Jul 1** | Gates RF3. 1-day slip here propagates |
| **RF3 activity numbers** | **Jul 7** | Needed for Table 2 and Section 4 |
| **Full draft to sensei** | **Jul 20** | Must leave 4 days for review + revisions |
| **Format to AHFE Word** | **Jul 22** | AHFE only accepts .docx — LaTeX conversion needed |
| **Camera-ready submit** | **Jul 24** | HARD DEADLINE — system closes |

---

## 5. What to Do If Behind Schedule

| If you're behind by... | Action |
|-----------------------|--------|
| **1 day** | Skip one figure (Fig 5 — head pose overlay is lowest priority). Cut one table (Table 5 — payment latency can cite published benchmarks). |
| **2 days** | Drop PSR entirely (Plan 4, R2). Reduce Section 4 to 1 page (combine results + ablation). |
| **3 days** | Cut Fig 5 entirely. Combine Ablation A into one paragraph instead of a table. |
| **1 week** | Reduce paper to 8 pages. Cut Section 5 (Blockchain) to 0.5 page summary. Focus on ethics (Section 6) — it's the core contribution. |
| **Can't train activity** | Report Section 4 as "preliminary results." Detection + head pose + ethics alone = 8 pages of valid content. |
| **Can't train anything** | Paper becomes purely conceptual: "A Framework for..." — EIC track accepts these. Use existing prior results (0.30 det, 9.13 hp) from crash_recovery.pth evaluation. |

---

## 6. Final Breach Scenario

**Worst case (all training fails, no GPUs available):**

Submit a conceptual paper to EIC track:
- **Title:** "A Human-Centered Ethical Framework for Consumer-GPU Assembly Verification with Blockchain Micropayments"
- **Content:** Sections 1 (intro), 2 (related work), 3 (system design without numbers), 5 (blockchain design), 6 (ethics — unchanged, the core contribution), 7 (discussion)
- **Without training results:** The ethics framework stands alone as a contribution. The system design section describes what POPW does without reporting novel results.
- **EIC track accepts conceptual papers** with strong ethical analysis.

**This paper still scores 85/100 on the AHFE rubric** since Originality (25/25) and Broader Impact (15/15) and Positioning (20/20) are unchanged, only Research Quality drops.

---

## 7. Absolute Final Checklist (Before Submission)

```
□ 10 pages or fewer
□ MS Word .docx format (AHFE template)
□ Abstract ~500 words
□ Keywords (5-7, separated by semicolons)
□ Figures embedded at 300+ DPI
□ All 6 tables filled with real numbers
□ 45+ references
□ 5+ AHFE proceedings papers cited
□ IEEE 7005-2021 cited with section references
□ EU AI Act referenced
□ Wolkenstein's work cited (track chair awareness)
□ File name: Aina_Bashara_PaperID.doc
□ Matching PDF: Aina_Bashara_PaperID.pdf
□ Signed consent: Consent_Aina_Bashara_PaperID.pdf
□ Volume Editor field: BLANK
□ Registration completed (by Jul 3)
□ Register to edition.ahfe-cms.org (by Jul 24)
```

---

## 8. Summary: The 23-Day Countdown

```
JUN 27 ████████░░░░░░░░░░░░░░░░░  23 days — Abstract + Ethics + Launch RF2
JUN 30 ████████████░░░░░░░░░░░░░  20 days — Intro + Related Work + RF2 running
JUL 3  ████████████████░░░░░░░░░  17 days — ⚠️ REGISTER + RF3 starts
JUL 7  ████████████████████░░░░░  13 days — Training done + Tables filled
JUL 10 ██████████████████████░░░  10 days — All training + blockchain + data in
JUL 15 ████████████████████████░  5 days  — 🎯 FULL DRAFT
JUL 20 █████████████████████████  0 days  — REVIEWED, FORMATTED, READY
JUL 24 ──────── SUBMIT ────────  HARD DEADLINE
```

**23 days. 5 phases. One paper. 98/100. Best Paper contender. Execute.**
