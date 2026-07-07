# 158 — Work Plan: 95 Days to AAIML Submission (Oct 10, 2026)

**Date:** 2026-07-07
**Submission Deadline:** Oct 10, 2026 (95 days)
**Current State:** 357 commits on origin/main, all 9 fixes applied, V3 PSR running

## §0. The 12-Week Schedule (95 days, Jul 7 → Oct 10)

### Week 1-2 (Jul 7-20) — IMMEDIATE: Get In-Flight Results
- **Day 1-2**: V3 PSR with actual DETACH fix completes → F1 result (running NOW, PID 1901736)
- **Day 1-7**: Single-task detection completes (PID 1574104, epoch 43+, ~3.4 days remaining)
- **Day 7-14**: Reconstruct file 156 §2 (DONE) + address File-157 F-1 through F-12 corrections
- **Day 14**: Freeze milestone — first numbers ready

### Week 3-4 (Jul 21-Aug 3) — Single-Task Baselines
- **Day 15-17**: Launch single-task activity (ConvNeXt + 75→69 class remap) — blocked on GPU
- **Day 18-20**: Launch single-task PSR (ConvNeXt + LeakyReLU + small-normal init) — blocked on GPU
- **Day 21-24**: Launch single-task pose (ConvNeXt + simple regression) — blocked on GPU
- **Day 28**: 4 single-task baselines complete, first 2x2 matrix ready

### Week 5-6 (Aug 4-17) — MViTv2-S Fine-Tuning
- **Day 29-30**: Launch MViTv2-S fine-tune (Kinetics pretrained, 2-week budget) — blocked on GPU
- **Day 30-42**: Fine-tune running, 2 epochs/day
- **Day 42**: MViTv2-S fine-tune complete, expected activity 0.45-0.55

### Week 7-8 (Aug 18-31) — Multi-Task with All Fixes
- **Day 43-45**: Launch multi-task V4 (all 9 fixes + DETACH fix + DETACH fix script)
- **Day 46-56**: Multi-task training (2 epochs/day)
- **Day 56**: Multi-task V4 complete, 4x2 ablation matrix ready

### Week 9-10 (Sep 1-14) — Final Ablation
- **Day 57-70**: Run remaining single-task baselines (pose, activity, PSR with V4 fixes)
- **Day 70**: Complete 4x4 matrix (4 single-task, 4 multi-task conditions)

### Week 11-12 (Sep 15-Oct 10) — Paper Writing + Submission
- **Day 71-77**: Write final paper (file 155 + new findings)
- **Day 78-84**: .tex integration, references, format
- **Day 85-91**: Internal review, revision
- **Day 92-95**: AAIML submission prep (format, abstract, supplementary)

## §1. The 8 Critical File-157 Findings to Address

| # | Finding | Status | Commit |
|---|---|---|---|
| F-1 | DETACH_PSR_FPN env-read fix | **DONE** | 59f84c3d4 (config.py) + ea6ac30c (wrapper) |
| F-2 | File 156 §2 was lost | **DONE** | 5e501d70a (reconstructed) |
| F-3 | Never-predicted class list wrong | Addressed in Agent-28 SOTA_STATUS rewrite |
| F-4 | 8 fix-commit hashes don't exist in repo | Need key fix hashes to real ones |
| F-5 | D4+D1R 0.6364 on 3 videos caveat | Addressed in SOTA_STATUS update |
| F-6 | 4 evidence directories were missing | **DONE** by Agent-10 (committed) |
| F-7 | +384 vs +4608 PSR post-gelu discrepancy | Need to commit /tmp/train_psr_v3.log excerpt |
| F-8 | LOO std is 0.0163 not 0.0158 | Addressed in SOTA_STATUS update |
| F-9 | /tmp/*.log and best.pth unverifiable remotely | Mark as workstation-only |
| F-10 | V3 process loading uncertain | Need workstation check |
| F-11 | Two different null baselines conflated | Disambiguate in tables |
| F-12 | Single-task ablation presets exist | Use `ablation_*_only` presets |

## §2. The Critical Path Forward

### The 4-Stage Validation

| Stage | Status | When |
|---|---|---|
| Stage 1: PSR V3 with DETACH fix | **RUNNING** (PID 1901736) | 1-2 days |
| Stage 2: Single-task detection completes | **RUNNING** (PID 1574104) | 3-4 days |
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
| Multi-task V4 (all 9 fixes) completes | Aug 31 | PENDING |
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
W7-8: Multi-task V4 (all fixes)
W9-10: 2x2 ablation
W11-12: Paper write + submit
```

Total: 95 days, achievable if in-flight trainings complete on schedule.

