# Opus Overview Prompt — POPW: Unified Egocentric Assembly Understanding

Paste the entire block below to Opus. It contains everything needed to understand the project's current state and what we need.

---

```
## PROJECT: POPW (Pass-Only Procedural Watch)
**Thesis**: A single shared-backbone model (ResNet-50 + FPN, 53M params) can perform egocentric assembly understanding — detection, body pose, head pose, activity, and procedure-step recognition — in one forward pass, at a fraction of the parameters and compute of separate specialists, without catastrophic interference, with cross-task FiLM conditioning.

**Paper**: `popw_paper_improved.tex` (~81 KB, ~1310 lines, >30 \todo/\popwres placeholders)
**Venue target**: Realistic — WACV/BMVC/workshop (given hardware constraints); aspirational — CVPR
**Hardware**: 1× RTX 3060 12GB, 64GB RAM (single GPU, commodity)

---

## CURRENT TRAINING STATE (as of 2026-06-22 12:00 UTC)

**Run 3** — Training crashed during epoch 21 and restarted from epoch 17 best checkpoint.

| Aspect | Detail |
|--------|--------|
| Stage | rf2 (50% subset, det + pose + head_pose heads active) |
| Current | epoch 17/36, PID 1204133, batch 2050/3302, ~58% through epoch |
| Best mAP50 | 0.2024 (from checkpoint — epoch 17 val) |
| Best mAP50_pc | 0.3036 (present-class only, 16/24 classes in 50% subset) |
| MAE | 9.13° (excellent — head pose is not a problem) |
| Batch speed | ~1.48s/it, 0.6-0.7 batch/s |
| GPU mem | 1.34GB / 12GB |
| Remaining budget | ~19 epochs × ~86 min = ~27h max (at max_epochs=36) |

**Run history:**
- **Run 1** (wrong LR/BIAS=4.0/2.0): epochs 17-21, mAP50=0.202-0.209, flat
- **Run 2** (correct LR/BIAS=1.0/1.0): epochs 17-21, mAP50=0.202-0.209, IDENTICALLY flat
- **CRASH** during epoch 21 (cause unclear — OOM? DataLoader?)
- **Run 3** (current): restarted from epoch 17 best checkpoint, generating new trajectory

**CRITICAL FINDING**: Run 1 (wrong LR/BIAS) and Run 2 (correct LR/BIAS) produce NEARLY IDENTICAL mAP50 trajectories across ALL 4 overlapping epochs. The mAP50 ceiling at ~0.207 is structural, not config-dependent.

| Epoch | Run 1 (2× LR, 4× Bias) | Run 2 (1× LR, 1× Bias) | Delta |
|-------|------------------------|------------------------|-------|
| 17 | 0.2039 | 0.2039 | 0.0000 |
| 18 | 0.2065 | 0.2065 | 0.0000 |
| 19 | 0.2088 | 0.2091 | +0.0003 |
| 20 | 0.2047 | 0.2069 | +0.0022 |

---

## WHAT WE KNOW (Confirmed)

1. **Structural ceiling at ~0.207 mAP50** — confirmed across 2 independent runs with 4× LR/BIAS difference. The ceiling is not config-dependent.
2. **LR restart at epoch 20 has ZERO effect** — regardless of base LR. Confirmed in both runs.
3. **Anchor coverage is fine** — POS_ANCHOR_PROBE shows 364-783 positive anchors/image (400-800 consistently). The "13-pos-anchor" fear was a pure overfit artifact from a 50-image test.
4. **Head pose is working** — MAE=9.13° (target is ≤60°). No issues.
5. **Detection dilution is real** — mAP50_pc=0.304 vs headline mAP50=0.207 (~50% higher). 8/24 classes have zero GT in the 50% subset, diluting the metric.
6. **Gradient norms healthy** — det ALIVE (1.24e+00), pose ALIVE (1.56e+00), head_pose ALIVE but borderline (4.47e-03). Frozen heads (act, psr) correctly DEAD.
7. **rf_stage_state.json IS writing** — heartbeats, det_health_history, checkpoint data all persisting correctly.

## WHAT WE HYPOTHESIZE (Primary Hypothesis)

**OHEM + FocalLoss gradient suppression** is the primary bottleneck causing the detection ceiling. Evidence:
- 50-image overfit shows a 3-regime trajectory: fast drop (1-5) → plateau (5-55 epochs, cls_loss ~0.3-0.4) → slow decline (55-200). The 55-epoch plateau is consistent with gradient suppression.
- Two independent main-training runs (Run 1, Run 2) both plateau at ~0.207 regardless of config.
- Changing LR/BIAS by 4× produces NO trajectory change — consistent with gradient-suppressed equilibrium.
- LR restart at epoch 20 has ZERO effect — consistent with near-zero gradients.

**CRITICAL**: This is still a hypothesis, not proven. Only an OHEM ablation (disable OHEM, keep FocalLoss, 5 epochs) can prove causation.

---

## WHAT THE PAPER NEEDS

The publishable bar (from GUIDE 4) is NOT beating SOTA. The thesis is:
> "A single shared-backbone model can perform egocentric assembly understanding in one forward pass without catastrophic interference."

Four things are needed:
1. **All 5 heads produce non-trivial results** (partially done — detection at 0.304 pc, head pose at 9.13°, activity never trained, PSR never trained, body pose never trained on POPW)
2. **Efficiency measurement** (53M params / 1 forward pass — measurable now)
3. **Ablation A** — single-task vs multi-task on identical backbone (NOT RUN)
4. **Ablation B** — FiLM conditioning contribution (NOT RUN)

Hardware bottleneck: 1× RTX 3060 12GB. ~27h remaining in rf2. ~15 epochs in rf3 at 35% subset. Phase B/C decoupled training implemented but not tested. IKEA ASM second dataset: 371 videos, ~2-3 days estimated.

---

## KEY OPEN QUESTIONS FOR OPUS

### Detection: Breaking the 0.207 Ceiling
1. Is OHEM ablation the right next step? Should we disable OHEM, keep FocalLoss, run 5 epochs?
2. If OHEM ablation fails (still ~0.20-0.22), what's the next experiment? Architecture limit? Label noise? Data quality?
3. 12/24 classes ALWAYS have AP=0 — the same 12, every run, every config. What causes exactly half the classes to be dead? Is this an anchor assignment issue (sizes 96-768 start too large for small assembly parts)?
4. Per-class AP is binary (0 or working) — no class has AP=0.05. Is this consistent with an anchor-matching threshold where a class either gets positive assignments or doesn't?

### Paper Strategy
5. Given our detection ceiling (0.207 mAP50, 0.304 pc), head pose is our strongest result. Should we lead with head pose + efficiency and frame detection as a fine-grained-state-discrimination problem?
6. Should we run the two ablations NOW (before detection is fixed) so we have a complete paper structure, or wait for detection improvement?
7. What venue is realistic? WACV? BMVC? Workshop? What threshold makes CVPR worth attempting?

### Activity & PSR (Never Trained)
8. Activity head was never trained. rf3 would enable it but requires mAP50≥0.40 gate. We're at 0.207. Is there a path to train activity without passing the rf3 gate?
9. PSR head (procedure-step recognition) was never trained. What's the minimal experiment to get a non-trivial F1?

### Resource Allocation
10. Given 1× GPU, ~27h remaining in rf2, ~15 epochs rf3, what's the optimal allocation of compute to maximize paper completeness? Should we skip rf3 entirely and go straight to ablations?

---

## FILES AVAILABLE (all in consult_2026_06_10/)

- `45_CURRENT_TRAINING_STATE.md` — Single source of truth for current state
- `46_DEEP_UNANSWERED_QUESTIONS.md` — Questions we may never answer
- `47_HYPOTHESES_PROVEN_WRONG_AND_UNANSWERED.md` — Full hypothesis catalog
- `50_ASK_OPUS_PAPER_PATH_TO_BENCHMARKABLE_RESULTS.md` — All paper-gap questions
- `GUIDE_4_THE_PAPER.md` — The publishable bar definition
- `code/` — Current source code (stage_manager.py, train.py)
- `logs/` — Latest training log, metrics.jsonl
- `evidence/` — rf_stage_state.json, eval metrics
```
