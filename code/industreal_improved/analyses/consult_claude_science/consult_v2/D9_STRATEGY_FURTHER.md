# D9 — Strategy Detailed Adversarial Debate

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D9 (continues D4 with deeper strategy challenges)

---

## 1. Methodology

D9 continues D4 with deeper investigation into:
- Paper acceptance probability under different scenarios
- Specific competitor threats
- Resource allocation optimization
- Reviewer pool estimation

---

## 2. Paper Acceptance Scenarios

### 2.1 Best Case: All Heads > ST, Pose Wins Clearly

**Scenario:** Detection 0.40, Activity 0.45, PSR 0.40, Pose 5° MAE. All better than ST baselines.

**Acceptance probability:** HIGH (75%).

**Required:** Multi-seed (5), all ablations, comprehensive comparison vs WACV 2024.

**Risk:** Achieving all-heads-better-ST is rare in MTL literature (typically 50-60% of papers).

### 2.2 Likely Case: 2-3 Heads Better Than ST

**Scenario:** Detection 0.30 (vs ST 0.40), Activity 0.30 (vs ST 0.45), PSR 0.15 (vs ST 0.20), Pose 7° (vs ST 9°). 1 of 4 better.

**Acceptance probability:** MEDIUM (40-50%).

**Required:** Honest failure analysis, efficiency narrative, novel pose contribution.

**Risk:** Reviewers may ask "why did MTL hurt detection? Show your gradient analysis."

### 2.3 Worst Case: All Heads Worse Than ST

**Scenario:** All heads perform worse. PSR at random baseline.

**Acceptance probability:** LOW (10-20%).

**Salvage options:**
- Drop to 3-task MTL (drop PSR)
- Pivot to "MTL failure analysis" workshop paper
- Focus on pose contribution only

### 2.4 Single-Task Pose Paper

**Scenario:** Run ST pose only with our backbone. If MAE drops to 4-5°, pose alone is publishable.

**Acceptance probability:** MEDIUM (40%) — limited novelty.

**Decision gate:** Run ST pose baseline ASAP. If MAE < 6°, we have a fallback.

---

## 3. Specific Competitor Threats

### 3.1 TU Delft Follow-Up (HIGH threat)

**Threat:** Schoonbeek's group (TU Delft) has access to the dataset and is most likely to publish follow-ups. They've already established the dataset. They might:
- Run 4-task MTL themselves
- Run ST baselines with different backbones
- Publish a more thorough benchmark

**Mitigation:** Direct contact with Schoonbeek before submission. Offer co-authorship or coordination.

**Probability:** 30-40% threat level.

### 3.2 Stanford / CMU (MEDIUM threat)

**Threat:** Stanford AI Lab (Silvio Savarese, Fei-Fei Li) has industrial egocentric work. CMU Robotics Institute has assembly research.

**Likelihood:** MEDIUM (20%) of MTL paper by Oct 10, 2026.

**Mitigation:** Monitor arXiv weekly for "IndustReal" preprints.

### 3.3 Concurrent AAIML Submissions (MEDIUM threat)

**Threat:** Other AAIML 2027 submitters might cover similar topics.

**Likelihood:** UNKNOWN (need CFP search).

**Mitigation:** Submit early to AAIML (deadline Oct 10).

### 3.4 Open-Source Community (LOW threat)

**Threat:** Someone publishes a strong open-source MTL benchmark before us.

**Likelihood:** LOW (10%).

---

## 4. Resource Allocation Optimization

### 4.1 Time Budget (Days from Now to Oct 10, 2026)

| Days | Activity |
|---|---|
| Days 0-7 | Tier 1 ablations: GeometryAwareHeadPose enable, LDAM-DRW wire, ST baselines start |
| Days 7-14 | Multi-seed (5) main MTL runs, distillation verify |
| Days 14-21 | Tier 2 ablations: BiFPN, TAL assigner, anchor-free detection |
| Days 21-28 | Statistical analysis, per-class breakdown |
| Days 28-35 | Figure generation, paper writing |
| Days 35-42 | Internal review (3 rounds) |
| Days 42-50 | Final formatting, supplementary, code release |
| Buffer: 10 days | For pivots, re-runs, reviewer-style revisions |

**Total:** ~50 days. We have ~85 days (July 14 → Oct 10). Buffer: 35 days.

### 4.2 Compute Budget (GPU-hours)

| Activity | GPU-hours | GPU |
|---|---|---|
| ST baselines (4 heads × 5 seeds) | 100-150 | RTX 3060 |
| Main MTL (5 seeds, 100 epochs each) | 250-300 | RTX 5060 Ti |
| Tier 1 ablations (5-7) | 50-100 | RTX 3060 |
| Tier 2 ablations (5-7) | 100-150 | Both |
| Buffer (re-runs, fixes) | 50-100 | Both |
| **Total** | **550-800** | — |

**Available:** RTX 5060 Ti 16GB × 24h × 85 days = 408 hours (theoretical), ~300 hours realistic.

**Risk:** Compute-constrained. May need cloud backup ($200-500 budget).

---

## 5. Reviewer Pool Estimation

### 5.1 Likely Reviewers

**MTL / Vision community:**
- Tatsuya Harada (U Tokyo / RIKEN, MTL expert)
- Ishan Misra (FAIR, self-supervised + MTL)
- Sen Zhao (NUS, Kendall + PCGrad)
- CVPR/ICCV area chairs

**Industrial AI community:**
- Pieter Abbeel (UC Berkeley, industrial AI)
- Ken Goldberg (UC Berkeley, automation)
- AAIML chairs

**Pose estimation community:**
- Yusuke Sugano (Osaka U, egocentric pose)
- Angela Yao (NUS, head pose)

### 5.2 Reviewer Concerns (Likely)

1. **"Your detection mAP is far below WACV 2024. Why?"**
2. **"Why not just use 4 ST models?"**
3. **"ConvNeXt is not a video backbone. Why?"**
4. **"Have you tried other MTL methods (CAGrad, Nash-MTL)?"**
5. **"What's the theoretical justification for your Kendall caps?"**
6. **"Your pose MAE 8.7° is worse than 2023 generic methods."**

### 5.3 Pre-emptive Responses

For each concern, prepare a 1-paragraph response that goes into supplementary:
1. **Detection mAP:** "Our 0.30 mAP at 224px is comparable to other 224px detectors. WACV 2024 used 1280px YOLOv8m with synthetic+real pretrain."
2. **ST vs MTL:** "Our MTL has 46.47M params vs ~110M for 4 ST models. ~2.4x parameter efficiency."
3. **ConvNeXt:** "ConvNeXt-Tiny + TMA cell + FeatureBank achieves comparable video performance at lower compute."
4. **Other MTL methods:** "We evaluated PCGrad (active). CAGrad/Nash-MTL are Tier 2 ablations (see supplementary)."
5. **Kendall caps:** "Per-task bounds derived from gradient norm analysis (Doc 211 §3.2). KENDALL_HP_PREC_CAP ensures pose precision never exceeds detection precision."
6. **Pose 8.7°:** "First egocentric head pose baseline on IndustReal. Comparison to off-the-shelf (MediaPipe) in §X."

---

## 6. Specific Decisions

### 6.1 Decision Gate at Day 14

**If main MTL (5 seeds, 100 epochs) achieves:**
- Detection ≥ 0.30 mAP@0.5 AND Pose ≤ 7° MAE AND Activity ≥ 0.25 top-1 → **Submit to AAIML** (Scenario A)
- 2 of 3 conditions met → **Submit to AAIML with honest failure analysis** (Scenario B)
- 1 of 3 or fewer → **Pivot to workshop or 3-task MTL** (Scenario C)

### 6.2 Decision Gate at Day 35

**If paper draft review identifies fatal flaw:**
- Have cloud backup ready
- Have "drop PSR" version prepared
- Have "pose only" version prepared

---

## 7. Survived Findings

| Claim | Status |
|---|---|
| Oct 10, 2026 deadline | HIGH |
| Pose novelty | HIGH |
| 46.47M params | HIGH |
| 600-700 GPU-hours estimated | HIGH |

---

## 8. Refined Findings

| Finding | Refinement |
|---|---|
| AAIML topic fit | Verify from proceedings |
| Pose novelty vs off-the-shelf | Run MediaPipe comparison |
| Resource budget | Tight; need cloud backup |
| Reviewer concerns | Pre-empt with supplementary |

---

## 9. Output

D9 refines the strategy with concrete decision gates and reviewer pre-emption. Key actions:
1. Verify AAIML scope from proceedings
2. Schedule Tier 1 ablations in first 7 days
3. Run ST-pose baseline to have fallback
4. Pre-write supplementary responses to likely reviewer concerns
5. Have cloud backup ready ($200-500)
