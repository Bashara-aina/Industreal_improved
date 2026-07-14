# FINAL IMPLEMENTATION PLAN — ULTIMATE Consultation V2

**Phase:** ULTIMATE Consultation V2 — Phase 3 Final Synthesis (Synthesizer S3)
**Date:** 2026-07-14
**Author:** Synthesizer S3
**Inputs:** FINAL_VERIFIED_FINDINGS.md + FINAL_RANKED_RECOMMENDATIONS.md

---

## Timeline Overview

**Start date:** 2026-07-14
**Submission deadline:** 2026-10-10 (AAIML 2027)
**Total days available:** 88
**Buffer:** ~30 days for pivots, re-runs, reviewer revisions

**Compute:** RTX 5060 Ti 16GB (Blackwell) + RTX 3060 12GB (Ampere), + cloud backup ($200-500 budget)

---

## Phase 1 — Tier 1 Foundation (Days 0-14)

### Day 0 (Today, 2026-07-14) — Setup + Quick Wins

**Morning (4 hours):**
- [ ] Verify AAIML scope (T1.10) — search proceedings
- [ ] Verify all module wiring (T1.9) — grep train.py for distillation, FAMO, RotoGrad, MetaBalance, LDAM-DRW, IMTL-L, TAL

**Afternoon (4 hours):**
- [ ] Re-measure gradient norms (T1.8) — run e8_gradient_diagnostic.py
- [ ] Run MediaPipe pose baseline (T1.7) — script + inference on test set

**Decision gate:** If MediaPipe MAE < 4°, our pose contribution needs reframing.

### Day 1-3 — Quick Wins + Enable Tier 1 Modules

**Tasks (in parallel where possible):**
- [ ] T1.1: Enable `USE_GEO_HEAD_POSE=1`, smoke test (0.5 day)
- [ ] T1.6: Uncapped Kendall ablation (1 day, RTX 3060)
- [ ] T1.8: Re-measure gradient norms (Day 0 finish)

**GPU allocation:** RTX 3060 for ablations, RTX 5060 Ti reserved for main runs

### Day 4-7 — ST Baselines Begin

**Schedule:**
- [ ] T1.3a: ST pose baseline (50 epochs × 5 seeds, RTX 3060) — 17.5 hours
- [ ] T1.3b: ST detection baseline (50 epochs × 5 seeds, RTX 3060) — 35 hours
- [ ] T1.3c: ST PSR baseline (50 epochs × 5 seeds, RTX 3060) — 25 hours
- [ ] T1.3d: ST activity baseline (50 epochs × 5 seeds, RTX 3060) — 25 hours

**Sequential on RTX 3060: ~100 hours = ~4-5 days wall-clock**

**Parallel on RTX 5060 Ti (idle during this phase):**
- [ ] T1.4: LDAM-DRW wire + smoke test (2 days)

**Decision gate (Day 7):**
- If ST detection mAP@0.5 ≥ 0.30 → continue to Phase 2
- If ST pose MAE > 15° → investigate code regression
- If ST activity top-1 < 5% → pose/PSR focus, drop activity

### Day 8-10 — Tier 1 Module Wiring

- [ ] T1.2: Distillation module verification + wiring (2 days)
- [ ] T1.4 finish: LDAM-DRW ablation run (if not already)

### Day 11-14 — Architecture Freeze + Pre-Main-MTL Ablations

- [ ] Freeze architecture: no major changes after Day 14
- [ ] Run final Tier 1 ablations: GeoHeadPose enabled, Kendall capped, PCGrad active
- [ ] Smoke test: 1-epoch main MTL run with all Tier 1 enabled

**Decision gate (Day 14):**
- Architecture frozen ✓
- All Tier 1 modules verified active ✓
- ST baselines complete ✓

---

## Phase 2 — Main Runs + Tier 2 Ablations (Days 14-42)

### Day 15-21 — Main MTL Multi-Seed Runs

- [ ] T1.5: Main MTL, 5 seeds × 100 epochs on RTX 5060 Ti
- [ ] Each seed: ~50-60 GPU-hours
- [ ] Sequential: 250-300 hours = 10-12 days wall-clock (24-hour operation)

**Parallel on RTX 3060:**
- [ ] T2.5: 2025-2026 literature search (background task, 1-2 days)
- [ ] T2.6: Confusion matrix analysis (0.5 day)

### Day 22-28 — Tier 2 Ablations

- [ ] T2.1: BiFPN swap (RTX 3060, 100 epochs, 1-2 days)
- [ ] T2.2: TOOD-TAL wiring (RTX 3060, 100 epochs, 2-3 days)
- [ ] T2.3: 480×480 resolution (RTX 5060 Ti if VRAM permits, 1-2 days)

### Day 29-35 — Statistical Analysis + Per-Class

- [ ] Bootstrap CIs on all metrics (10000 resamples)
- [ ] Paired bootstrap tests: MTL vs ST (4 heads)
- [ ] Bonferroni correction: p < 0.05/4 = 0.0125
- [ ] Per-class breakdown: detection (24 cls), activity (75 cls), PSR (11 components)

**Decision gate (Day 35):**
- All multi-seed results available
- All Tier 2 ablations evaluated
- Statistical significance established

---

## Phase 3 — Paper Writing (Days 35-65)

### Day 36-42 — First Draft

**Sections to write (in order of importance):**
- [ ] Abstract (300 words)
- [ ] Introduction (1.5 pages)
- [ ] Method: Architecture (1 page)
- [ ] Method: Training (1 page) — Kendall caps, PCGrad, distillation
- [ ] Experiments: Setup (0.5 page)
- [ ] Experiments: Main Results (1 page, Table 3 from V1 doc 224)
- [ ] Experiments: Ablation (1 page, Table 5)
- [ ] Experiments: Efficiency (0.5 page, Table 6)
- [ ] Related Work (0.5 page)
- [ ] Conclusion (0.5 page)

### Day 43-49 — Internal Review

- [ ] Round 1: Claim verification (all numbers vs raw data)
- [ ] Round 2: Figure quality (resolution, fonts, colors)
- [ ] Round 3: Narrative coherence

### Day 50-56 — Revisions

- [ ] Address review feedback
- [ ] Re-run any verification experiments
- [ ] Add supplementary material

### Day 57-65 — Code Release + Reproducibility

- [ ] Public GitHub repo
- [ ] README, requirements, reproduce.sh
- [ ] Zenodo DOI for checkpoint archival
- [ ] Pretrained weights on HuggingFace

---

## Phase 4 — Final Review + Submission (Days 65-88)

### Day 66-75 — External Review

- [ ] Send draft to 2-3 external reviewers
- [ ] Wait for feedback
- [ ] Incorporate feedback

### Day 76-85 — Final Polish

- [ ] Address external feedback
- [ ] Re-verify all claims
- [ ] Camera-ready formatting

### Day 86-88 — Submission

- [ ] AAIML 2027 submission portal
- [ ] Supplementary material upload
- [ ] Code release confirmation

**Buffer: 3 days before Oct 10 deadline**

---

## Decision Gates Summary

| Day | Gate | Decision Criteria |
|---|---|---|
| 0 | Setup complete | AAIML scope verified ✓ |
| 7 | ST baselines complete | ST det mAP ≥ 0.30, pose MAE < 15° |
| 14 | Architecture frozen | All Tier 1 modules active, smoke test passes |
| 21 | Main MTL 50% complete | Composite score on trajectory |
| 35 | All multi-seed done | Bootstrap CIs, statistical tests |
| 49 | First draft complete | All sections written |
| 65 | Code released | Repo + Zenodo + HF ready |
| 88 | Submitted | AAIML portal confirms |

---

## Risk Mitigation

### Hardware Failure
- **Risk:** RTX 5060 Ti or RTX 3060 fails
- **Mitigation:** Cloud backup ($200-500 budget). RunPod RTX 4090 at $0.34/hr.

### Code Regression
- **Risk:** Architecture freeze broken by unexpected bug
- **Mitigation:** Smoke test after every change. Rollback capability via `--disable-X` flags.

### Timeline Slip
- **Risk:** Day 14 architecture freeze not met
- **Mitigation:** Drop Tier 2 ablations, focus on Tier 1 only. Submit 3-task MTL if 4-task fails.

### Pose MAE > MediaPipe
- **Risk:** Off-the-shelf beats our contribution
- **Mitigation:** Reframe as "first MTL pose baseline on IndustReal" (not "SOTA pose"). Or pivot to "MTL comparison" with pose as auxiliary.

### AAIML Scope Mismatch
- **Risk:** AAIML is not industrial AI focused
- **Mitigation:** Pivot to WACV or ICRA. Earlier deadlines possible.

---

## Compute Allocation Summary

| Phase | RTX 3060 (hours) | RTX 5060 Ti (hours) | Cloud (hours) | Total |
|---|---|---|---|---|
| Phase 1 (Days 0-14) | 150 | 50 | 0 | 200 |
| Phase 2 (Days 14-42) | 100 | 280 | 0 | 380 |
| Phase 3 (Days 35-65) | 0 | 0 | 0 | 0 |
| Phase 4 (Days 65-88) | 0 | 0 | 50 (emergency) | 50 |
| Buffer | 50 | 50 | 100 (emergency) | 200 |
| **Total** | **300** | **380** | **150** | **830** |

**Available:** RTX 5060 Ti 16GB × 24h × 88 days = 528 hours theoretical, ~380 realistic.
RTX 3060 12GB × 24h × 88 days = 396 hours theoretical, ~300 realistic.

**Risk:** Tight. Cloud backup essential.

---

## Critical Path

```
Day 0:  Setup + Verify AAIML + MediaPipe
Day 7:  ST baselines done
Day 14: Architecture freeze
Day 35: All multi-seed complete
Day 49: First draft
Day 88: Submitted
```

**Float:** ~3 days at end of timeline.

---

## Per-Task Schedule

### Pose Head
- Day 0: T1.7 (MediaPipe baseline)
- Day 1: T1.1 (GeoHeadPose enable)
- Day 4: ST pose baseline
- Day 35: Pose results available

### Detection Head
- Day 4-7: ST detection baseline
- Day 22-28: T2.2 (TOOD-TAL), T2.3 (480px)
- Day 35: Detection results

### Activity Head
- Day 4-7: ST activity baseline
- Day 8-10: T1.4 (LDAM-DRW)
- Day 22-28: T2.6 (confusion matrix)
- Day 35: Activity results

### PSR Head
- Day 4-7: ST PSR baseline
- Day 1-3: T1.6 (uncapped Kendall)
- Day 35: PSR results

---

## Output

This file is the day-by-day implementation plan. S4 (paper framework) should reference the Day 36-42 draft structure. S5 (Claude Science queries) should address the open verification questions raised here.
