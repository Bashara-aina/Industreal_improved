# 110 — Contribution Audit: Top-10 Novelty Claims Under Hostile Review

**Date:** 2026-07-03
**Method:** Every claim checked against (a) the actual repo code/config, (b) live literature search. Includes corrections of the consultant's OWN prior statements.

## Verdict Table

| # | Claim | Verdict | Required repair before submission |
|---|---|---|---|
| 1 | Interface-mismatch pathology catalogue (F1/F13/F18/F22 + sampler×bank + scheduler cadence) | **CONFIRMED** — but "taxonomy of silent DL bugs" already exists (Humbatova ICSE'20; Tambon EMSE'23; SANER'24 PyTorch). Delta = composition-level failures between individually-correct mechanisms in MTL training, with mechanism+probe+fix | (a) Call it "case-study catalogue + detection methodology", not taxonomy. (b) **Run A/B replays for F1 and F18** (2 epochs, bug on/off, same seed) — current before/after is confounded (F1/F4/F7/F8 landed in one restart) |
| 2 | First head-pose baseline on IndustReal (8.92° fwd) | **CONFIRMED but RECATEGORIZED** — pose.csv is the WEARER's head pose: this is egocentric ego-pose regression, NOT face head-pose estimation. OpenFace/6DRepNet comparisons (docs 98/106) are category errors — remove. "SOTA-competitive" is meaningless (no prior SOTA); only "first reported" is true | Full-test-split converged eval; note epoch-5 val = raw weights, epoch-8+ = EMA (deltas not purely learning) |
| 2b | "16.6mm position — excellent" | **REJECTED as stated** — rests on HEAD_POSE_POS_SCALE=100 heuristic ("cm-scale plausible"); doc 85 itself says DO NOT REPORT. HoloLens typically exports meters | Verify unit against official IndustReal release or drop the number |
| 3 | Single-pass 4-task system on consumer GPU | **CONFIRMED as system, DOWNGRADED as novelty** — must position vs EgoT2 (CVPR'23), EgoPack (CVPR'24), IMPACT/MECCANO. Scope: "first single-pass single-backbone ASD+action+PSR+ego-pose ON INDUSTREAL". Efficiency axis currently UNMEASURED (FPS/FLOPs all \todo); "31% fewer params" is 29.3%; "$299 GPU" false for 5060 Ti 16GB ($429 MSRP; $299 = 3060) | Measure FPS/FLOPs; fix arithmetic; run Ablation A |
| 4 | Passenger/driver task coupling (PSR isolation) | **DOWNGRADED** to supporting experiment inside #3 — frozen-feature transfer is well-trodden; delta is concurrent in-run isolation (runtime-verified zero-grad) + controlled flip (not yet run) | Run the RF6 detach flip |
| 5 | Head-pose FiLM conditioning helps activity | **REJECTED as current contribution** — zero evidence (Ablation B never run); gaze-conditioned egocentric AR is established literature. Valid hypothesis only | Run Ablation B; claim only if significant |
| 6 | PSR evaluation integrity package | **CONFIRMED** — fill-forward gaming, psr_pos weakness, F22 forensics all verified. CORRECTION: "random F1≈0.14" was measured at synthetic density (11 transitions/40 frames) — NOT transferable; recompute null baseline on real sequences before quoting | Recompute null baseline; frame vs STORM-PSR as efficiency finding, never accuracy contender |
| 7 | MTL observability toolkit | **CONFIRMED as artifact** — value is interpretation rules (lv*=ln L, cap fossils, OHEM score reading), not the plumbing | Release as artifact with paper #1 |
| 8 | Kendall-in-practice findings | **CONFIRMED as section** — 4 code-verified findings; cite scalarization-vs-MTL literature | Fold into paper #1 |
| 9 | Verb-grouping protocol | **CONFIRMED, smallest** — breaks MViTv2 comparability; paper must either re-eval MViTv2 under grouping or drop the comparison row | Grouping-none ablation exists; run it |
| 10 | Honest-benchmarking methodology | **CONFIRMED** — combined-metric 81%-saturation case study verified in code; matched-sampler ablation design implemented | None |

## Revised ranking
1 > 2 > 3 > 6 > 10 > 4 > 8 > 7 > 9; #5 exits until Ablation B exists.

## Three fact-fixes gating ANY submission
1. Resolve the position-unit question (or delete every mm claim).
2. Delete OpenFace/6DRepNet; reposition head pose as egocentric ego-pose regression.
3. A/B replays for F1 and F18 so pathology-cost claims are causal, not anecdotal.

## Sources
IndustReal WACV 2024 (arXiv:2310.17323); github.com/TimSchoonbeek/IndustReal; STORM-PSR (arXiv:2510.12385); Humbatova et al. ICSE 2020; Tambon et al. EMSE 2023 (silent bugs Keras/TF); SANER 2024 PyTorch silent bugs; EgoT2 CVPR 2023 (arXiv:2212.06301); EgoPack CVPR 2024 (arXiv:2403.03037); IMPACT (arXiv:2604.10409).
