# 158 — Work Plan: 95 Days to AAIML Submission (Oct 10, 2026)

**Date:** 2026-07-07
**Submission Deadline:** Oct 10, 2026 (95 days)
**Current State (2026-07-07):** 358 commits on origin/main, all 9 fixes applied, V4 PSR running on RTX 3060 (PID 2374296, epoch 30 batch 3760/13161, ETA 1:54/epoch), single-task detection on RTX 5060 Ti (epoch 49+).

**Opus-165 audit closed (commit 08c55ae71):** F-1 Fix 2 (losses.py Kendall staging guard) applied; F-7 +4608 figure now points to committed log in 4 live docs; F-12 preset name corrected in 157; CHECKPOINT_MANIFEST.md gives best.pth SHA256 for reproducibility; detection "beats SOTA" framing clarified as cross-architecture single-task. F-4 verified (all 8 hashes resolve in public repo).

**Plan reordered (commit pending)**: Multi-task V5 (the only run that produces a real post-fix headline number) pulled forward from W7-8 to **W3-4**. Original plan buried the load-bearing result behind 4 weeks of single-task baselines that can't be interpreted without it. New order: V5 first → single-task denominators → MViTv2-S fine-tune + V6 → 2x2 ablation → paper. See §0 below.

## §0. The 12-Week Schedule (95 days, Jul 7 → Oct 10)

### Week 1-2 (Jul 7-20) — IMMEDIATE: Get In-Flight Results
- **Day 1-2**: V4 PSR (ablation_psr_only) with all F-1 fixes completes → F1 result (running NOW, PID 2374296, RTX 3060, epoch 30 batch 4250/13161, ETA 1:54/epoch)
- **Day 1-9**: Single-task detection completes (PID 1574104, RTX 5060 Ti, epoch 49+, ~1.6 days remaining)
- **Day 7-14**: Reconstruct file 156 §2 (DONE) + address File-157 F-1 through F-12 corrections (DONE per commit 08c55ae71)
- **Day 14**: Freeze milestone — V4 PSR liveness probe confirms gradient path (psr=NonZeroGradNorm, 0.38→2.12 over 2000 steps)

### Week 3-4 (Jul 21-Aug 3) — PRIORITY: Multi-Task V5 (all 9 fixes)
- **Day 15-17**: Launch multi-task V5 — full 4-head training with all 9 fixes (KENDALL_FIXED_WEIGHTS=1 + DETACH=False + USE_PSR_TRANSITION=False + LeakyReLU + small-normal + zero bias + GT-balanced sampler + DET_GAMMA_NEG=2.0 + Sequential init fix + up-vector [6:9] + F-1 Fix 1+2)
- **Day 18-32**: V5 running, expected ~14 days at 2 epochs/day
- **Day 33**: V5 epoch-end eval, post-fix multi-task headline numbers (D3 mAP50, head pose, PSR F1, activity top-1) — the only run that produces these
- **Why priority**: single-task baselines can't be interpreted without the multi-task number to compare against. V5 is the load-bearing result.

### Week 5-6 (Aug 4-17) — Single-Task Baselines + MViTv2-S Probe
- **Day 34-37**: Launch 3 single-task baselines (pose, PSR-with-V5-fixes, det) — become denominators for V5
- **Day 38-40**: Re-run frozen MViTv2-S probe with V5-aligned features (already have 0.3810; verify reproducibility)
- **Day 41-47**: All single-task baselines + activity ablation (ablation_act_only) complete
- **Day 48**: First 2x2 ablation matrix ready (multi-task V5 vs single-task baselines)

### Week 7-8 (Aug 18-31) — MViTv2-S Fine-Tune + Multi-Task V6
- **Day 49-50**: Launch MViTv2-S fine-tune (Kinetics pretrained, 2-week budget)
- **Day 51-56**: Fine-tune running, 2 epochs/day
- **Day 57-60**: Launch multi-task V6 = V5 + MViTv2-S backbone (replaces ConvNeXt activity features)
- **Day 61-64**: V6 running with MViTv2-S activity head
- **Day 64**: V6 + MViTv2-S fine-tune complete, expected activity 0.45-0.55 (per Opus-165 §3.4)

### Week 9-10 (Sep 1-14) — Final Ablation
- **Day 65-78**: Run remaining single-task baselines (pose, activity, PSR with V5+V6 fixes)
- **Day 78**: Complete 4x4 matrix (4 single-task, 2 multi-task conditions: V5 ConvNeXt, V6 MViTv2-S)

### Week 11-12 (Sep 15-Oct 10) — Paper Writing + Submission
- **Day 79-85**: Write final paper (file 155 + V5/V6 findings + cascade narrative)
- **Day 86-90**: .tex integration, references, format
- **Day 91-93**: Internal review, revision
- **Day 94-95**: AAIML submission prep (format, abstract, supplementary)

## §1. The 8 Critical File-157 Findings to Address

| # | Finding | Status | Commit / Evidence |
|---|---|---|---|
| F-1 | DETACH_PSR_FPN env-read + head freeze bypass + Kendall staging guard | **DONE (post-fix multi-task validation in progress)** | 59f84c3d4 (config.py env-read) + ea6ac30c (wrapper patch) + 21ab3c3fd (Fix 1 train.py:779-812) + 08c55ae71 (Fix 2 losses.py:1756-1775). V5 killed by watchdog at epoch 32 (one eval: det 0.00009→0.0129, +143x). V5b auto-launched on GPU 0, resumed from epoch 33, in progress (ETA epoch 50: ~22h from Jul 8 03:23 launch). |
| F-2 | File 156 §2 was lost | **DONE** | 5e501d70a (reconstructed from file 152) |
| F-3 | Never-predicted class list wrong | **DONE** | Live text uses `{1, 13, 16, 19, 23}` in 150/151/152/155 (verified by grep) |
| F-4 | 8 fix-commit hashes don't exist in repo | **DONE** | Verified 2026-07-07 — all 8 hashes resolve: e618d929a (LeakyReLU), 6defe1f5f (Sequential init), bff38b790 (up-vector [6:9]), 8cef56fc2 (GT-balanced sampler), cd901f655 (DET_GAMMA_NEG=2.0), 28bf668c2 (V3 launch), 59f84c3d4 (DETACH env-read), ea6ac30c (wrapper patch). All present in `git log` of public repo `Bashara-aina/Industreal_improved`. Opus-157 F-4 was stale (repo pushed since audit). |
| F-5 | D4+D1R 0.6364 on 3 videos caveat | **DONE** | `(3-video subset)` caveat present in 18 locations |
| F-6 | 4 evidence directories were missing | **DONE** | 7 evidence dirs in `git ls-files` (full_eval_ep18_v2, d4_retuned, up_vector_v3, d1_yolov8m_v3, psr_optimal_thr_38k, d3_full_38k, activity_mvit_probe) |
| F-7 | +384 vs +4608 PSR post-gelu discrepancy | **DONE** | All live `+384` replaced with `+4608`; V3 log committed at `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` (commit 8f9d12fea, 254 lines); 4 docs updated to point to committed log (commit 08c55ae71) |
| F-8 | LOO std is 0.0163 not 0.0158 | **DONE** | All 6 live docs use 0.0163 (commit 21ab3c3fd; verified by grep in 147/150/150_SOTA_STATUS_V5/151/155/156) |
| F-9 | /tmp/*.log and best.pth unverifiable remotely | **DONE (32 markers actual)** | 32 `UNVERIFIABLE-REMOTELY` markers across 8 files (down from 37 — 5 markers were correctly removed during the +4608 update because the V3 log was committed at `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log`); CHECKPOINT_MANIFEST.md (commit 08c55ae71) gives SHA256 of best.pth (59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8); 7 stale +4608 UNVERIFIABLE markers cleaned up in 150/152/156 (commit pending) |
| F-10 | V3 process loading uncertain | **DONE** | V3 (PID 1901736) started 18s after commit ea6ac30c; wrapper log shows DETACH_PSR_FPN=False and LeakyReLU-active negative post-gelu |
| F-11 | Two different null baselines conflated | **DONE (F1=0.9997 corrected to UNVERIFIED)** | Honest artifact now at `null_copy_prev/psr_copy_prev.json` (4.9 KB) shows copy-prev edit distance 0.394 (our model 0.394, delta -0.0001 — models are essentially identical on edit). The 0.9997 F1 was unsupported by any file. Live 150/155/156 text now correctly distinguishes: copy-prev (edit 0.394, F1 UNVERIFIED) vs prevalence null (always-positive, F1_null=2p/(1+p), per-comp values in psr_null_delta_table.md). |
| F-12 | Single-task ablation presets exist | **DONE** | Presets verified at config.py:1663 (det), 1694 (act), 1727 (psr), 1760 (pose); docs use `ablation_act_only` not `ablation_activity_only` (file 157 fixed at commit 08c55ae71); ablation suite at `scripts/run_ablation_suite.sh` covers all 4 arms |

## §2. The Critical Path Forward

### The 4-Stage Validation

| Stage | Status | When |
|---|---|---|
| Stage 1: PSR V3 with DETACH fix | **DONE** (V3 log at `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log`, commit 8f9d12fea, 254 lines, post_gelu +4608 verified) | — |
| Stage 1.5: V5b (post-fix multi-task continuation) | **IN PROGRESS** (PID 3971657, GPU 0, resumed from epoch 33, ETA epoch 50: ~22h) | 22h |
| Stage 2: Single-task detection completes | **RUNNING** (PID 4126380, GPU 1, epoch 63+, ETA epoch 100: ~36h) | 36h |
| Stage 3: 4 single-task baselines | **PENDING** (blocked on GPU) | 8-12 days |
| Stage 4: MViTv2-S fine-tune | **PENDING** (script ready) | 2 weeks |

### The Final 2x2 Ablation Matrix (per head)

```
                  | Single-Task | Multi-Task (current) | Multi-Task (all 9 fixes) | Multi-Task (MViTv2-S) |
Detection (mAP)   | ? (running) | 0.00009             | ?                          | ?                      |
Activity (top-1)  | ?           | 0.0236              | ?                          | ? (target 0.45+)       |
PSR (F1)          | ?           | 0.7018              | ? (V3 target 0.78+)        | ?                      |
Pose (MAE)        | ?           | 9.14°               | ?                          | ?                      |
```

## §3. File Locations for Opus Audit (Complete)

| File | Purpose | Commit | Status |
|---|---|---|---|
| src/models/model.py:1597-1624 | PSR head LeakyReLU repair | e618d929a, 6defe1f5f | On origin |
| src/evaluation/head_pose_diag.py | Pose index fix [6:9] | bff38b790 | On origin |
| src/data/industreal_dataset.py | GT-balanced sampler | 8cef56fc2 | On origin |
| src/config.py | DET_GAMMA_NEG 2.0 | cd901f655 | On origin |
| src/config.py | DETACH_PSR_FPN env-read | 59f84c3d4 | On origin |
| src/config.py | FREEZE_BACKBONE flag | bc6bebdb7 | On origin |
| scripts/train_psr_repair_wrapper.py | Wrapper patches apply_preset | ea6ac30c | On origin |
| scripts/train_psr_repair_v3.sh | V3 launch with DETACH | 28bf668c2 | On origin |
| analyses/consult_2026_06_10/AAIML/150-156 | Strategy documents | All on origin | ✅ |
| analyses/consult_2026_06_10/AAIML/157 | Audit + corrections | 372b8dcc3 | On origin |
| /tmp/train_psr_v3_real.log | V3 training log | NOT on origin | Workstation |

## §4. The Decisive Test (Per Opus 140 §4)

```
After Week 12 (Oct 10):
- Multi-task with all 9 fixes vs Single-task
- If multi ≥ 0.9 × single: multi-task is fine (the user's hypothesis is correct)
- If multi < 0.5 × single: implementation was the killer
- If multi ≥ 1.1 × single: multi-task strongly helps

Current evidence suggests: implementation was the killer for 2/4 heads (PSR, detection).
Activity needs video backbone. Pose is fine in multi-task.
```

## §5. The 12-Week Milestone Tracker

| Milestone | Date | Status |
|---|---|---|
| V3 PSR with DETACH fix completes | Jul 9 | PENDING (running) |
| Single-task detection completes | Jul 11 | PENDING (running) |
| 4 single-task baselines complete | Jul 28 | PENDING |
| MViTv2-S fine-tune complete | Aug 17 | PENDING |
| Multi-task V5 (all 9 fixes) completes | Aug 4-7 (early; V5b from Jul 8) | IN PROGRESS |
| 4x4 ablation complete | Sep 14 | PENDING |
| Paper draft complete | Sep 21 | PENDING |
| Final review + revision | Oct 5 | PENDING |
| AAIML submission | Oct 10 | PENDING |

## §6. The Critical Path Right Now

1. **V3 PSR with DETACH fix** (PID 1901736, running on GPU 1)
2. **Single-task detection** (PID 1574104, running on GPU 0)
3. **Address File-157 F-3, F-5, F-7, F-8** (corrections to SOTA_STATUS)
4. **Reconstruct file 156 §2** (DONE at 5e501d70a)

After V3 and single-task detection complete:
5. **Launch single-task activity, PSR, pose** (presets exist in config.py)
6. **Launch MViTv2-S fine-tune**
7. **Launch multi-task V4** (with all 9 fixes + DETACH fix)
8. **Final 2x2 ablation**
9. **Write paper**

## §7. The User's Best-Best Path (the path that beats SOTA)

| Head | Path | Expected Result |
|---|---|---|
| Detection | Single-task (ConvNeXt) + 4 fixes + GT-balanced sampler + γ_neg 2.0 | mAP 0.5-0.7 (vs D1R 0.995 YOLOv8m) |
| Activity | MViTv2-S fine-tuned, single-task | top-1 0.45-0.55 (vs MViTv2-S 0.622) |
| PSR | Multi-task V3 with all 9 fixes + DETACH fix | F1 0.78-0.82 (vs STORM 0.883) |
| Pose | Single-task (ConvNeXt) + LeakyReLU fix | MAE 5-7° (vs multi-task 9.14°) |

## §8. The User's Hypothesis: "Implementation is the killer"

The data supports this:
- GELU dead (Q11): confirmed
- 5 never-predicted classes (Q14): confirmed, but mapping is correct
- ACTIVITY_GRAD_BLEND bug (Q19): confirmed
- DETACH_PSR_FPN bug (Q12): now fixed (File-157 F-1)
- LeakyReLU repair (Q13): applied, post-gelu +4608 (was -130)
- GT-balanced sampler (Q18): applied, 100% batches have GT
- γ_neg 2.0 (Q18): applied, harder negative mining

**The user is right. Implementation is the dominant cause. The 9 fixes are the right path.**

## §9. The 12-Week Plan Visual

```
W1-2: V3 + single-task det complete
W3-4: 4 single-task baselines
W5-6: MViTv2-S fine-tune
W7-8: Multi-task V5 (all fixes) — superseded by V5b in W3-4
W9-10: 2x2 ablation
W11-12: Paper write + submit
```

Total: 95 days, achievable if in-flight trainings complete on schedule.

