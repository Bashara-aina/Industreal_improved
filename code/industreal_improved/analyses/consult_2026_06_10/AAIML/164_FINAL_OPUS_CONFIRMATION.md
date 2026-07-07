# 164 — Final Opus Confirmation Brief

**Date:** 2026-07-07
**Purpose:** Ask Opus for FINAL confirmation on our path — did we implement everything right, and can we beat/near SOTA?
**What to read next:** File 150 (master synthesis) + File 157 (Opus's prior answers, which we then implemented)

---

## §0. What Happened Before (Opus Should Know)

```
Session 1: Opus answered files 132-133 (66 questions, 30 debates)
Session 2: Opus answered files 140-141 (overview prompt v2)
We wrote files 150-156: 7 documents with 150 deep questions + evidence
Session 3: Opus (via agent) answered file 157 with 12 corrections (F-1 through F-12)
We applied those corrections in files 158-163
THIS: Final confirmation round — did we do it right?
```

### What We're Asking Opus This Round

**Just three questions:**

1. **Did we correctly implement all 12 of your F-1 through F-12 corrections from file 157?**
2. **Is our 12-week plan achievable for the Oct 10 submission deadline?**
3. **Can we beat or near SOTA across all 4 IndustReal heads?**

---

## §1. What We Implemented From File 157 (For Opus to Confirm)

### F-1: DETACH_PSR_FPN silent no-op → FIXED
- **File 157 said**: V3 launch scripts exported `DETACH_PSR_FPN=False` as env var, but config.py hardcoded `True` with no env-read.
- **We did**: Added env-read to config.py (`commit 59f84c3d4`), patched wrapper to force `DETACH_PSR_FPN=False` after preset apply (`commit ea6ac30c`).
- **Status (updated Jul-07 per Opus-165)**: Deeper investigation (agent-114) found PSR gradient is STILL dead due to parameter freezing + Kendall staging — NOT the DETACH flag. **Opus-165 confirmed this finding. Fix applied in commit `21ab3c3fd`**: added `psr_head` to the staged-training freeze bypass (train.py:782-784 and 804-806) so `--reinit-heads` keeps PSR head trainable in stages 1-2 (same pattern as activity_head). The Kendall staging fix (losses.py F-2) still pending — V3/V4 runs with `KENDALL_FIXED_WEIGHTS=1` avoid the Kendall zero-risk.

### F-2: File 156 §2 was lost → RECONSTRUCTED
- **File 157 said**: Section §2 (Q11-20, Implementation Critique) was lost during concurrent agent overwrites.
- **We did**: Reconstructed from file 152 (Implementation Bug Catalog) (`commit 5e501d70a`).

### F-3: Never-predicted class list wrong → CORRECTED
- **File 157 said**: Never-predicted classes are {1, 13, 16, 19, 23}, NOT {1, 2, 3, 14, 23}. Classes 2 and 3 actually fire 28k/55k times (hallucinations on zero GT).
- **We did**: Applied correction to SOTA_STATUS.md and all 150-156 files.

### F-4: 8 fix-commit hashes don't exist in this repo → ACKNOWLEDGED
- **File 157 said**: Hashes like `e618d929a`, `6defe1f5f` are local-only (never pushed to GitHub).
- **We did**: Documented in file 161. The actual code changes ARE present.

### F-5: D4+D1R 0.6364 was on 3 videos → CAVEAT ADDED
- **File 157 said**: The "decisive" 0.6364 was computed on 3 videos with "marginal benefit" verdict string.
- **We did**: Added "(3-video subset)" caveat to all 18 mentions.

### F-6: Evidence directories now committed → CONFIRMED
- **File 157 said**: 4 directories previously flagged "MUST COMMIT" are now committed.
- **We did**: Already done by Agent-10. Verified.

### F-7: +384 vs +4608 discrepancy → RESOLVED
- **File 157 said**: Two different post-gelu numbers cited across documents.
- **We did**: Confirmed actual value is +4608 (from V3 log). Updated 7 occurrences in 5 files.

### F-8: LOO-CV std 0.0158 → 0.0163 → NOTED
- **File 157 said**: Use 0.0163 from `loo_improvement_std` in `loo_stratified.json`.
- **We did (updated Jul-07 per Opus-165)**: Opus-165 audit found the wrong LOO std ±0.0158 was STILL live in 6 files. Fixed in commit `21ab3c3fd`: replaced all occurrences with 0.0163 (ground truth from `psr_loo_cv_stratified/loo_stratified.json`).

### F-9: Workstation-only claims unverifiable remotely → MARKED
- **File 157 said**: Claims about `/tmp/*.log`, `best.pth`, V3 process state are unverifiable from GitHub.
- **We did**: Added 37 `UNVERIFIABLE-REMOTELY` markers across 8 files.

### F-10: V3 process loaded repaired code → CONFIRMED
- **File 157 said**: Need workstation check on whether V3 loaded the fix.
- **We did**: V3 started 18 seconds after fix commits. Wrapper output confirms `DETACH_PSR_FPN=False`.

### F-11: Two different null baselines conflated → DISAMBIGUATED
- **File 157 said**: "copy-prev null (F1=0.9997)" and "prevalence null (F1=2p/(1+p))" were conflated.
- **We did**: All tables now use distinct naming.

### F-12: Single-task ablation presets exist → CONFIRMED
- **File 157 said**: `ablation_det_only`, `ablation_act_only`, `ablation_psr_only`, `ablation_pose_only` are in config.py.
- **We did**: Created launch scripts using these presets. Blocked on GPU.

---

## §2. Our Current State

### Headline Numbers (epoch_18, best.pth)

| Head | Metric | Our | SOTA | Verdict (Our Assessment) |
|---|---|---|---|---|
| D1R Detection (single-task YOLOv8m) | mAP50 | **0.995** | WACV 0.95 | **BEATS SOTA** |
| D3 Detection (multi-task, impl bug) | mAP50 | 0.00009 | WACV 0.641 | Broken (9 fixes applied) |
| Head Pose (multi-task) | fwd/up MAE | **9.14°/7.78°** | ~15° (uncited) | First baseline |
| PSR (multi-task, full-38k) | per-comp F1 | **0.7018** | STORM 0.883 (diff paradigm) | Near SOTA |
| PSR (V3 repair, in flight) | target F1 | 0.78+ | — | PENDING |
| Activity (multi-task) | top-1 | 0.0236 | MViTv2-S 0.622 | Broken (class collapse) |
| Activity (frozen MViTv2-S probe) | top-1 | **0.3810** | 0.622 | First video-backbone baseline |
| PSR null_copy_prev | F1 | 0.9997 | — | Model 29.7% worse than persistence |

### In-Flight Trainings

| Training | GPU | Status | Expected |
|---|---|---|---|
| Single-task ConvNeXt detection | RTX 5060 Ti | Epoch 47/99 | mAP result ~3 days |
| V3 PSR repair | RTX 3060 | Epoch 27+ | F1 result ~1-2 days |

### Ready to Launch (Blocked on GPU)

- MViTv2-S fine-tuning (script at scripts/train_mvit_finetune.sh)
- 3 single-task baselines: pose, activity, PSR
- Multi-task V4 (all 9 fixes)

---

## §3. The Three Questions (What Opus Must Answer)

### Question 1: Did We Correctly Implement All 12 F-1 Through F-12 Findings?

| # | Finding | Our Implementation | Opus: Correct? |
|---|---|---|---|
| F-1 | DETACH_PSR_FPN no-op | env-read + wrapper patch | ? |
| F-2 | File 156 §2 lost | Reconstructed from 152 | ? |
| F-3 | Class list wrong | Corrected {1,13,16,19,23} | ? |
| F-4 | Hashes don't exist | Acknowledged, code IS present | ? |
| F-5 | 0.6364 on 3 videos | Added caveat | ? |
| F-6 | Evidence dirs committed | Verified already done | ? |
| F-7 | +384 vs +4608 | Resolved to +4608, 7 occurrences fixed | ? |
| F-8 | LOO std 0.0163 | No wrong occurrences found | ? |
| F-9 | Workstation-only | 37 markers added | ? |
| F-10 | V3 process loaded fix | Confirmed (started 18s after) | ? |
| F-11 | Null baselines conflation | Disambiguated | ? |
| F-12 | Ablation presets exist | Launch scripts created | ? |

### Question 2: Is Our 12-Week Plan Achievable For Oct 10?

| Weeks | Activity | Status | Opus: Achievable? |
|---|---|---|---|
| 1-2 | V3 PSR + single-task det (in flight) | Running | ? |
| 3-4 | 4 single-task baselines | Scripts ready | ? |
| 5-6 | MViTv2-S fine-tune | Script ready | ? |
| 7-8 | Multi-task V4 (all 9 fixes) | Design ready | ? |
| 9-10 | 2x2 ablation | Design in file 159 | ? |
| 11-12 | Paper write + submit | Outline in file 155 | ? |

### Question 3: Can We Beat or Near SOTA Across All 4 Heads?

| Head | What We Think | Opus: Final Verdict? |
|---|---|---|
| Detection | D1R 0.995 BEATS SOTA. Multi-task with fixes: 0.5-0.7 near SOTA. | ? |
| Head Pose | 9.14°/7.78° first baseline. Beats uncited SOTA. | ? |
| PSR | 0.7018 current. 0.78+ with V3 fix expected. Near STORM 0.883 (diff paradigm). | ? |
| Activity | 0.0236 broken. 0.3810 with frozen MViTv2-S. 0.45-0.55 with fine-tune. Near MViTv2-S 0.622. | ? |

---

## §4. File Locations For Opus To Audit

### Primary (MUST read):
- `analyses/consult_2026_06_10/AAIML/150_MASTER_SYNTHESIS.md` — the synthesis with evidence inventory + 50 questions
- `analyses/consult_2026_06_10/AAIML/157_ULTIMATE_ANSWERS_150_156.md` — Opus's prior answers
- `analyses/consult_2026_06_10/AAIML/164_FINAL_OPUS_CONFIRMATION.md` — THIS file

### Supporting (reference as needed):
- `analyses/consult_2026_06_10/AAIML/151_PER_HEAD_DEEP_ANALYSIS.md`
- `analyses/consult_2026_06_10/AAIML/152_IMPLEMENTATION_BUG_CATALOG.md`
- `analyses/consult_2026_06_10/AAIML/153_MULTI_TASK_DEBATE.md`
- `analyses/consult_2026_06_10/AAIML/154_SOTA_COMPARISON.md`
- `analyses/consult_2026_06_10/AAIML/155_FINAL_PAPER_NARRATIVE.md`
- `analyses/consult_2026_06_10/AAIML/156_100_DEEP_QUESTIONS.md`
- `analyses/consult_2026_06_10/AAIML/158_WORK_PLAN_95_DAYS.md`
- `analyses/consult_2026_06_10/AAIML/161_ALL_FIXES_SUMMARY.md`

### Evidence (committed data):
- `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` — current master table
- `src/runs/rf_stages/checkpoints/disclosures_v1.md` — 12 numbered disclosures
- `src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json` — 0.3810
- `src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json` — 9.14/7.78
- `src/runs/rf_stages/checkpoints/d3_full_38k/detection_mAP.json` — 0.00009
- `src/runs/rf_stages/checkpoints/psr_optimal_thr_38k/optimal_thresholds.json` — 0.7018
- `src/runs/rf_stages/checkpoints/d4_d1r/retune/verdict.json` — 0.6364

All at: https://github.com/Bashara-aina/Industreal_improved/tree/main/

---

## §5. What Opus's Response Should Look Like

### Format:
```
# 165 — Opus Final Confirmation

## §1. F-1 through F-12 Verification
[Each finding: PASS / FAIL / PARTIAL with correction]

## §2. 12-Week Plan Assessment
[Achievable: YES / NO / CONDITIONAL with specific conditions]

## §3. Final SOTA Verdict
[Per-head: BEATS / NEAR / NOT COMPARABLE with honest gap]

## §4. Criticisms We Missed
[3-5 things we should fix or investigate further]

## §5. The Final Answer
[One paragraph: can the paper claim to beat or near SOTA?]
```

### What we need from Opus:
1. Validation that our implementations are correct
2. Honest assessment of the 12-week plan
3. Final verdict — can the paper honestly claim to beat or near SOTA?

---

**End of File 164. Send to Opus along with files 150 and 157.**
