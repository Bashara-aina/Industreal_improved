# Opus RF4 Consultation — Master Index

**Date:** 2026-07-02
**Current PID:** 554646 (epoch 2/99, batch 830/4387)
**Goal:** Pass Gate RF4 → RF10 → AAIML paper with SOTA-comparable results
**Post-Agent-Review Corrections:** Applied inline — see each file for CORRECTION markers

## File Map

| File | Topic | Key Question for Opus |
|---|---|---|
| [89-index-and-contents.md](89-index-and-contents.md) | Master index with TL;DR and known issues | Start here for context |
| [90-training-status-trajectory.md](90-training-status-trajectory.md) | Current run state, complete RF4 history, loss trajectories across epochs 0→1→2 | Does the loss trend indicate healthy convergence? |
| [91-architecture-deep-dive.md](91-architecture-deep-dive.md) | Full architecture: ConvNeXt-Tiny backbone, 5 heads, TMA cell, temporal bank, Kendall weighting | Is there a fundamental architectural issue? |
| [92-loss-analysis-multitask-balancing.md](92-loss-analysis-multitask-balancing.md) | Per-head loss formulas, magnitudes, Kendall interaction, known vs unknown | Is multi-task balancing correct? |
| [93-validation-metrics-gate-status.md](93-validation-metrics-gate-status.md) | Validation history (zero successful runs), step-val data, expected targets, gate criteria | Are we on track to pass RF4? |
| [94-fixes-paper-targets-strategic-recommendations.md](94-fixes-paper-targets-strategic-recommendations.md) | All 22 fixes, AAIML strategy, required baselines, contingency plans | What should we change architecturally? |
| [95-50-deep-questions-for-opus.md](95-50-deep-questions-for-opus.md) | 50 probing questions on architecture, loss, detection, activity, training infrastructure | Companion deck for deeper Opus review |
| [96-FABLE-RF4-CONSULTATION-ANSWER.md](96-FABLE-RF4-CONSULTATION-ANSWER.md) | **THE ANSWER (2026-07-02)** — code-verified findings (incl. critical seq-batch grad-wipe bug F1), answers to all 7 questions + 50-question triage, 14 fixes applied on branch, restart protocol, AAIML strategy | Read this after 89 |
| [97-current-status-deep-analysis.md](97-current-status-deep-analysis.md) | **DEEP STATUS (2026-07-03)** — 8h44m training history, ALL 24 fixes verified, loss trends epochs 2→4, Kendall progression, probability assessment for RF4→RF10 | Current state after fixes |
| [98-head-by-head-analysis.md](98-head-by-head-analysis.md) | **PER-HEAD ANALYSIS** — each head's architecture, trajectory, fix impact, expected convergence timeline, conditions for SOTA-benchmarkable results | Which heads are viable? |
| [99-aaiml-viability-benchmarking.md](99-aaiml-viability-benchmarking.md) | **AAIML VIABILITY** — gap analysis to publishable thresholds, risk quadrants per head, fallback tiers A-E, decision matrix | Can we win AAIML? |
| [100-deep-20-questions-for-opus.md](100-deep-20-questions-for-opus.md) | **20 NEW QUESTIONS (2026-07-03)** — GPU stability crisis, detection/activity, PSR/pose, multi-task balancing, infrastructure | New questions after implementing F1-F16 |
| [101-overview.md](101-overview.md) | **MASTER OVERVIEW** — navigation guide for all 12 files, reading order, critical path | How to read the package |
| [102-FABLE-ROUND5-20-ANSWERS.md](102-FABLE-ROUND5-20-ANSWERS.md) | **ROUND 5 ANSWERS (2026-07-03)** — all 20 questions answered with premise corrections (audit-agent files described a different codebase!), GPU crisis playbook (Xid/Xorg diagnostic ladder, TF32 mitigation was backwards), lv_pose=-1.000 solved, combined-metric unit bug, F19-F21 | Read after 100 |

## TL;DR for Opus

**Training:** Alive at epoch 2/99 on RTX 5060 Ti with batch=6. Detection class count is 24. Activity output is 75 raw classes, grouped to ~41-47. Effective batch is 48 (paper target 16) without LR adjustment — likely slowing all heads.

**Losses:** Pose converged (8.38→0.1) but may be magnitude-matching, not directional. Detection stable (~2.0). Activity flat (~1.0, only at 40% ramp). PSR near-zero on non-seq batches (structurally zero due to transition objective).

**Validation:** Zero successful full evals. Two step-vals showed activity improving (1→3 classes, entropy 0→1.036 nats). Detection LOCALIZING with good boxes but scores stuck at bias init (score_p50=0.036 vs sigmoid init ~0.033).

**Critical concerns discovered post-agent-review:**
- Kendall log_vars are NEVER logged — cannot verify multi-task balancing
- OneCycleLR pct_start: code uses 0.3 (peak epoch 31), docs say 0.1 (peak epoch 10) — 3x discrepancy
- PSR contributes ZERO backbone gradient due to detach_psr_fpn=True — multi-task thesis undermined for this head
- Activity gets ~20x less gradient signal than PSR on seq batches (activity ~0.9 vs PSR ~10.0)
- Detection/backbone gradient ratio is ~0.001 — detection is a free rider on other heads' features

## What We Want From Opus

1. **LR scaling & scheduler** — Should effective batch 48 (3x paper) have proportional LR increase?
2. **Kendall diagnostics** — How to log and interpret log_var values?
3. **Activity gradient** — Should ACTIVITY_LOSS_WEIGHT increase from 1.0 to 5.0+?
4. **PSR gradient isolation** — Should detach_psr_fpn be disabled in later stages?
5. **Detection score separation** — Is FocalLoss(gamma=2) suppressing positive gradient at bias init?
6. **Ablation strategy** — Which ablations are mandatory for AAIML acceptance?
7. **Contingency planning** — When to activate Tier 1/2/3 fallback plans?

## Agent-Review Identified Gaps (Post-Hoc Corrections)

**CORRECTION: Architecture gaps filled:** Detection classes corrected (24 not 63), activity outputs corrected (75 raw not 69), TMA Cell 0-param claim questioned, FPN gradient isolation flagged, effective batch mismatch documented.

**CORRECTION: Loss analysis gaps filled:** Kendall log_vars missing diagnostic added, PSR structured-zero gradient documented, activity gradient starvation quantified, pose magnitude-matching caveat added, combined metric sensitivity flagged.

**CORRECTION: Strategy gaps filled:** AAIML viability critique, required ablation studies, Tier 1/2/3 contingency plans, epoch 3 and 6 threshold tables.

## Known Discrepancies & Infrastructure

- **LR scaling:** EFFECTIVE_BATCH=48 (3x paper's 16) with no LR adjustment — linear scaling rule suggests 3x LR
- **OneCycleLR pct_start:** optimizer.py line 58 has pct_start=0.3 (peak epoch 31), docs say 0.1 (peak epoch 10)
- **Dual GPU:** RTX 3060 12GB idle — subprocess eval can be re-enabled
- **MIXED_PRECISION:** BF16 may work on RTX 5060 Ti (Blackwell) without GradScaler

## Verification Status

- [x] 22 documented fixes applied (19 original + 3 activity root cause fixes)
- [x] GPU stable (no OOM, no watchdog kills)
- [x] Batch=6 utilizing 7.5/16GB VRAM
- [x] Epoch 0 completed (first RF4 run ever)
- [x] Resumed from checkpoint to epoch 2
- [ ] LR scaling issue and pct_start discrepancy pending resolution
- [ ] First validation at epoch 3 (~1 hour away)
- [ ] Full detection mAP at epoch 6
- [ ] Convergence expected ~epoch 20-30
