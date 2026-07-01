# AAIML 2027 — 101-Day Execution Plan to Camera-Ready

**Deadline:** October 10, 2026 (101 days from July 1, 2026)
**Hardware:** 2× RTX 3060 12GB (GPU 1 primary, GPU 0 ablations)
**Current Status:** All code fixes applied, 85% confidence, training at RF4 epoch ~3.

---

## Phase 1: Foundation (July 1-15) — 15 Days

### Week 1 (July 1-7): Primary Training + Critical Ablations

| Day | GPU 1 (Primary) | GPU 0 (Ablations) |
|-----|-----------------|-------------------|
| Jul 1 | RF1 start (seed 42, --reinit-heads) | Setup verify |
| Jul 2 | RF1 continue → RF2 | Ablation 1: MLP vs TCN/ViT (2 conditions × 15 epochs) |
| Jul 3 | RF2 → RF3 | Ablation 1 continue |
| Jul 4 | RF3 → RF4 | Ablation 2: Balanced vs CB sampling (2 conditions × 15 epochs) |
| Jul 5 | RF4 (50% data, all heads) | Ablation 2 continue |
| Jul 6 | RF4 continue | Ablation 3: Kendall bounds 3 conditions × 18 epochs |
| Jul 7 | RF4 → RF5 | Ablation 3 continue |

**Checkpoint Jul 7:** 
- [ ] RF5 running
- [ ] All 3 critical ablations completing
- [ ] First real numbers available for paper

### Week 2 (July 8-15): Seeds + Factory Pilot Prep

| Day | GPU 1 | GPU 0 |
|-----|-------|-------|
| Jul 8-9 | RF5→RF6→RF7 | Seed 73 (RF1-RF3) |
| Jul 10-11 | RF7→RF8→RF9 | Seed 73 continue |
| Jul 12-14 | RF9→RF10 (100% data) | Seed 128 (RF1-RF3) |
| Jul 15 | **RF10 COMPLETE** (seed 42 done) | Seed 128 continue |

**Milestone Jul 15:** Primary 3-seed training ~60% complete. Start factory pilot (Phase 1).

---

## Phase 2: Factory Pilot (July 15 - August 1) — 17 Days

| Date | Activity |
|------|----------|
| Jul 15-17 | Setup: edge GPU installation, dashboard deployment, worker onboarding |
| Jul 18-31 | **Two-week pilot run** |
| Jul 25 | Mid-pilot check: opt-out rate, dashboard usage, initial SUS |
| Aug 1 | Pilot ends. Collect survey data. Conduct interviews. |

**If pilot misses this window:** Strip to 1 paragraph with "feasibility demonstrated; detailed results in future work." Do not delay paper for pilot.

---

## Phase 3: Supplementary Ablations (July 20 - August 1) — GPU Time Permitting

If critical path (Phase 1) completes early:

| Ablation | Conditions | Epochs | GPU Hours | Priority |
|----------|-----------|--------|-----------|----------|
| DET_GT_FRAME_FRACTION | 0.40, 0.60, 0.90, 0.0 | 20 | ~90 | Medium |
| 2×2 sub-grid (MLP vs TCN, T=4 vs 16) | 4 | 15 | ~90 | Low |
| GRAD_CLIP 1.0 vs 5.0 | 2 | 20 | ~45 | Low |
| WD 1e-3 vs 5e-2 | 2 | 20 | ~45 | Low |

**Total if all run:** ~270 GPU hours. Only run if seed training and critical ablations are complete.

---

## Phase 4: Writing (August 1 - September 1) — 31 Days

### Week 5-6 (Aug 1-14): Draft Core Sections

| Section | Owner | Pages | Due |
|---------|-------|-------|-----|
| Abstract + Introduction | — | 1.2 | Aug 7 |
| Related Work | — | 0.8 | Aug 7 |
| System Architecture (incl. hyperparameter table) | — | 1.0 | Aug 10 |
| Pathology 1 (broadened) | — | 1.0 | Aug 14 |
| Pathology 2 (corrected) | — | 0.7 | Aug 14 |
| Pathology 3 (shortened) | — | 0.5 | Aug 14 |

### Week 7-8 (Aug 14-28): Draft Remaining Sections

| Section | Pages | Due |
|---------|-------|-----|
| Infrastructure Lessons (new) | 0.3 | Aug 18 |
| Empirical Results (real numbers) | 1.0 | Aug 21 |
| Deployment + Pilot (trimmed) | 0.5 | Aug 25 |
| Limitations (new) + Conclusion | 0.8 | Aug 28 |

### Milestone Aug 28: First complete draft with real numbers.

---

## Phase 5: Figures + Supplementary (August 1 - September 15) — 45 Days

### Figures (3 mandatory)

| Figure | Description | Tool | Due |
|--------|-------------|------|-----|
| Fig 1 | Pathology 1 mechanism (dual-panel: data pipeline + optimization pipeline) | Matplotlib / Illustrator | Aug 15 |
| Fig 2 | Ablation results (MLP vs TCN/ViT, balanced vs CB, s_act trajectories, pred_distinct) | Matplotlib | Aug 20 |
| Fig 3 | Gradient artifact illustration (per-parameter norm ratio vs head-level RMS gradient) | Matplotlib | Aug 25 |

### Supplementary Material

| Item | Content | Due |
|------|---------|-----|
| Supplement A | Full blockchain architecture + smart contract code | Sep 1 |
| Supplement B | Ethics governance framework (IEEE 7005-2021 mapping table) | Sep 1 |
| Supplement C | Factory pilot thematic analysis + interview protocol | Sep 5 |
| Supplement D | 20-repo survey methodology + repository list | Sep 5 |
| Supplement E | Full training configuration (all 30+ hyperparameters) | Sep 10 |

---

## Phase 6: Polish + Code Release (September 1 - October 10) — 40 Days

### September 1-15: Internal Review

| Task | Duration |
|------|----------|
| First full draft read-through | 2 days |
| Check all cross-references (\Cref) | 1 day |
| Check all acronyms defined on first use | 1 day |
| Verify IEEE format compliance | 1 day |
| Fix overfull hboxes, page breaks | 2 days |
| Reviewer defense prep (v2) | 3 days |
| Simulated review session | 2 days |

### September 15-30: Code Release + Final Experiments

| Task | Duration |
|------|----------|
| Upload code to public GitHub (verify URL) | 1 day |
| Create README with reproduction instructions | 2 days |
| Verify single-command reproduction pipeline | 2 days |
| Run final reproducibility verification (fresh conda env → reproduce numbers) | 5 days |
| Generate DOI for code archival (Zenodo) | 1 day |

### October 1-10: Final Polish

| Task | Due |
|------|-----|
| Remove all \inprogress commands | Oct 1 |
| Final proofread | Oct 3 |
| Colleague review | Oct 5 |
| Final format check (PDF compliance) | Oct 7 |
| **SUBMIT** | **Oct 10** |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| RF1-RF10 produces activity top-1 < 10% | Medium | High | Fall back to 4 tasks, report activity as negative result |
| Factory pilot delayed past Aug 1 | High | Medium | Strip pilot to 1 paragraph, refer to future work |
| Three-seed variance too large for meaningful CIs | Medium | Medium | Report mean ± min/max range instead of bootstrap CI |
| Ablation 3 (Kendall bounds) shows no s_act divergence | Medium | Medium | Reframe Pathology 2 as theoretical + preemptive, not empirical |
| Code repo can't be made public (IP constraints) | Low | High | Use institutional repository with access-controlled review copy |
| IRB approval delayed | Low | **Critical** | Cannot submit without IRB. Escalate at Nihon University. |

---

## Pre-Submission Checklist (Final 48 Hours)

- [ ] No \inprogress macros in source
- [ ] All cross-references verified (\Cref, \ref)
- [ ] All acronyms defined on first use
- [ ] IRB protocol number present and correct
- [ ] Figures generated and included (\includegraphics)
- [ ] Table 2 populated with 3-seed mean + CI
- [ ] Code URL resolves to public repository
- [ ] Supplementary material compiled and linked
- [ ] Overfull hboxes fixed (0 warnings)
- [ ] IEEE format compliance (margins, font, header)
- [ ] PDF under 10 MB
- [ ] Acceptance confirmation from all co-authors
