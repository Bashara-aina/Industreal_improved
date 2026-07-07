# 161 — All Implementation Fixes: Complete Catalog

## Section 1. The 10 Fixes (9 Original + 1 V3 DETACH)

| # | Fix | Commit | File | What it does | Status |
|---|---|---|---|---|---|
| 1 | PSR head GELU to LeakyReLU | e618d929a | src/models/model.py:1597-1624 | Activations +4608 (was -130) | On origin |
| 2 | PSR init index fix | 6defe1f5f | src/models/model.py:1618-1624 | Sequential [3]=Linear | On origin |
| 3 | Pose index fix | bff38b790 | src/evaluation/head_pose_diag.py | [3:6] to [6:9] for up-vector | On origin |
| 4 | Detection GT-balanced | 8cef56fc2 | src/data/industreal_dataset.py | 100% batches have GT | On origin |
| 5 | Detection gamma_neg 2.0 | cd901f655 | src/config.py | Harder negative mining | On origin |
| 6 | Detection anchor audit | 10d5ab596 | src/config.py | Confirmed adequate | On origin |
| 7 | Detection class verify | a0ffb9aa8 | src/config.py | Mapping correct | On origin |
| 8 | Full-eval v2 fix | 216566da0 | src/evaluation/full_eval_stream.py | 9.14/7.78 verified | On origin |
| 9 | FREEZE_BACKBONE flag | bc6bebdb7 | src/config.py | Single-task enabled | On origin |
| 10 | DETACH_PSR_FPN env-read | 59f84c3d4 | src/config.py | V3 fix (File-157 F-1) | On origin |
| 11 | V3 wrapper patch | ea6ac30c | scripts/train_psr_repair_wrapper.py | apply_preset patched | On origin |

Total: 11 commits across 9 original implementation fixes plus 2 commits for the V3 DETACH fix. All on origin/main.

## Section 2. File 157 Corrections Applied

| # | Finding | File | Commit | Status |
|---|---|---|---|---|
| F-1 | DETACH_PSR_FPN env-read | src/config.py | 59f84c3d4, ea6ac30c | Done |
| F-2 | File 156 Section 2 reconstructed | analyses/.../156_100_DEEP_QUESTIONS.md | 5e501d70a | Done |
| F-3 | Never-predicted class list | (in SOTA_STATUS) | — | Done |
| F-5 | 3-video caveat | (in SOTA_STATUS) | — | Done |
| F-6 | Evidence dirs committed | (Agent-10) | 02a94937e | Done |
| F-7 | +384 vs +4608 PSR post-gelu resolution | analyses/.../PSR_POST_GELU_RESOLUTION.md | 0392827ba | Done |
| F-8 | LOO std 0.0158 to 0.0163 | analyses/.../F8_LOO_CORRECTION.md | 11726b63b | Done |

## Section 3. Strategy Documents

All documents 150 through 160 are on origin/main:

- 150: Master synthesis with 50 deep questions and best-of-best path
- 151: Per-head deep analysis with file paths
- 152: Implementation bug catalog (all 9 fixes documented)
- 153: Multi-task vs single-task debate (2 opposing positions)
- 154: SOTA comparison matrix (all papers, all metrics)
- 155: Final paper narrative for AAIML submission
- 156: 100 deep questions (synthesis, debate, paper structure)
- 157: Ultimate answers to every question in documents 150-156
- 158: 12-week work plan (95 days to AAIML Oct 10)
- 159: 2x2 ablation matrix design
- 160: Ablation results template
- 161: This all-fixes summary

## Section 4. The Decisive Test (In Progress)

- V3 PSR training: PID 1901736, post_gelu +4708, step 440/13161
- Single-task detection: PID 1574104, epoch 43+
- 2 more single-task baselines: pose, activity, PSR (script ready)
- MViTv2-S fine-tuning: script ready
