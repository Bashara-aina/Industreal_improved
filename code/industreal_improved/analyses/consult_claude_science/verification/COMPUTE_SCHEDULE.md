# COMPUTE SCHEDULE — GPU Allocation to Submission

**Budget (2026-07-14):** GPU 0 (RTX 3060 12GB) = **116 GPU-h** · GPU 1 (RTX 5060 Ti 16GB) = **365 GPU-h** · Total = **481 GPU-h**
**Throughput cap:** ~16 combined GPU-h/day (thermal/power) — GPU 1 ≈ 10 h/day, GPU 0 ≈ 6 h/day sustained
**Run costing:** main MTL 100-ep ≈ 50 GPU-h (GPU 1) · ablation 50-ep ≈ 25 GPU-h · ST per launcher: pose 3.5, det 7, act 5, psr 5 GPU-h per seed (RTX 3060, 50 ep)

## Master allocation table

| Day range | Dates | GPU 0 (RTX 3060) | GPU 0 h | GPU 1 (RTX 5060 Ti) | GPU 1 h | Cum. total |
|-----------|-------|------------------|---------|----------------------|---------|-----------|
| 1 | Jul 14 | ST smoke tests (all 4 heads, 1-ep) + launch ST pose ×3 | 2 + start | Main MTL baseline seed 42, 100 ep (start) | start | — |
| 1–2 | Jul 14–15 | ST pose ×3 seeds | 10.5 | baseline continues | — | — |
| 2 | Jul 15 | MediaPipe baseline eval | 2 | baseline continues | — | — |
| 2–4 | Jul 15–17 | ST activity ×3 seeds | 15 | baseline continues | — | — |
| 4–6 | Jul 17–19 | ST PSR ×3 seeds | 15 | baseline finishes Day 5 | 50 | 94.5 |
| 6–8 | Jul 19–21 | ST detection ×3 seeds | 21 | Ablation #1: UW-SO + per-task LR + EMA warmup (50 ep) | 25 | 140.5 |
| 9–11 | Jul 22–24 | Confusion matrix + evals (2) · cRT retrain if gated (5) | 2–7 | Ablation #2: BiFPN (or TSBN if det gate fired), 50 ep | 25 | ~172 |
| 12–14 | Jul 25–27 | figures/eval only | 1 | Ablation #3 (gated occupant: ASL / MetaBalance / OHEM / none) | 0–25 | ~198 |
| 12–16 | Jul 25–29 | — | — | (alt.) MViTv2-S full run if activity gate fired | 0–50 | — |
| 15–20 | Jul 28–Aug 2 | eval-pipeline dry runs | 1 | Final-config seed 42 (100 ep; or +25 h resume of ablation winner) | 25–50 | ~240 |
| 21 | Aug 3 | **ARCHITECTURE FREEZE** — ledger review: GPU 1 remaining must be ≥ 125 h | | | | |
| 22–24 | Aug 4–6 | per-seed evals, qual. figures | 2 | Final config seed 123 (100 ep) | 50 | ~292 |
| 25–30 | Aug 7–12 | efficiency measurements (3060 per protocol) | 2 | Final config seed 7 (100 ep) | 50 | ~344 |
| 31–35 | Aug 13–17 | — | — | Gated: distillation teachers + run (55) OR OHEM ablation (25) OR idle | 0–55 | ≤399 |
| 36–55 | Aug 18–Sep 7 | — | — | reserve (emergency re-runs) | 0 | — |
| 56–60 | Sep 8–12 | — | — | **Test-split eval, ONCE** (3 seeds) | 3 | ≤402 |
| 61–88 | Sep 13–Oct 10 | — | — | emergency only | 0 | ≤402 |

## Budget ledger

### GPU 0 (RTX 3060) — 116 h cap
| Item | GPU-h |
|------|-------|
| ST smoke (4×1-ep) | 2.0 |
| ST pose ×3 | 10.5 |
| ST activity ×3 | 15.0 |
| ST PSR ×3 | 15.0 |
| ST detection ×3 | 21.0 |
| MediaPipe baseline | 2.0 |
| Confusion matrix + misc evals | 3.0 |
| cRT activity retrain (gated) | 0–5.0 |
| Efficiency measurement | 2.0 |
| **Committed** | **70.5–75.5** |
| **Reserve** | **40.5–45.5** ✅ |

### GPU 1 (RTX 5060 Ti) — 365 h cap
| Item | GPU-h |
|------|-------|
| Main baseline seed 42 (100 ep) | 50 |
| Ablation #1 candidate config (50 ep) | 25 |
| Ablation #2 BiFPN/TSBN (50 ep) | 25 |
| Ablation #3 (gated; worst case MViT) | 0–50 |
| Final config seed 42 (resume path 25 / fresh 50) | 25–50 |
| Final config seed 123 | 50 |
| Final config seed 7 | 50 |
| Test-split eval | 3 |
| **Committed (base)** | **228–303** |
| Gated extras (distill 55 / FAMO 25 / OHEM 25 — at most one) | 0–55 |
| **Worst case** | **358** |
| **Reserve (typical path: ~253 committed)** | **62–137** ✅ |

## Feasibility checks

- **Worst case total:** 75.5 + 358 = 433.5 GPU-h < 481 ✅ (with hard rule: at most ONE of {MViT, distillation} may fire — both = +105 h and breaks the ledger; if both gates trigger, MViT wins, distillation → future work).
- **Calendar:** GPU 1 committed ≈ 253–358 h at 10 h/day = 26–36 GPU-days, scheduled across Days 1–35 ✅ (fits with weekends of slack). GPU 0 committed ≈ 75 h at 6 h/day = 13 days, scheduled Days 1–11 ✅.
- **Deadline margin:** all compute ends Day 60 (Sep 12); 28 days of pure writing buffer remain.

## Deltas vs the pre-verification plan (UNANSWERED_QUESTIONS Q50 draft)
1. **ST baselines: 5 seeds → 3 seeds** (102.5 → 63.5 h incl. smoke) — matches the project's own metrics protocol (seeds 42/123/7); 5 seeds would have consumed 88% of GPU 0 alone.
2. **Multi-seed MTL: 250–300 h → 125–150 h** (3 seeds, with a 25 h saving if the final config equals the ablation-#1 config and we resume from its epoch-50 checkpoint).
3. **The draft's "400 needed vs 300 available — TIGHT" is resolved:** worst-case 433.5 vs 481 available, typical-case ~330.

**End of COMPUTE_SCHEDULE.md**
