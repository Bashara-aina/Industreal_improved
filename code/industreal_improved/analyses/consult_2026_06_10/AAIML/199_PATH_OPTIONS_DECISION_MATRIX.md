# 199 — Path Options & Decision Matrix: Proving "MTL Helps, Not Hurts"

**Date:** 2026-07-10
**Status:** run11 LIVE (first eval at ep10 in ~6 hours)

---

## 0. The Central Question

> One shared MViTv2-S backbone does 4 tasks. Is MTL helping (positive transfer), not hurting (bounded cost), and more efficient than 4 specialists?

**Why this matters:** The Kendall-collapse diagnosis (our methodological contribution) is novel. But a paper needs experimental evidence. The experiment is: *MTL vs matched single-task baselines across 4 tasks, with per-head MTL/ST ratios.* The story can be:

- **Strong paper (Goal A+):** MTL > ST on ≥1 head (positive transfer), MTL ≈ ST on ≥2 heads (benign sharing), 1 head is a bounded cost. Efficiency win is clear.

- **Good paper (Goal A):** MTL is at 70-90% of ST on all heads. Bounded cost, efficiency win is clear, Kendall-collapse is the methodological contribution.

- **Weak paper:** MTL << 70% of ST on ≥2 heads. Efficiency claim becomes "we trade accuracy for lower inference cost" — weaker but still publishable as a trade-off study.

- **Unpublishable:** MTL at 0.0 on ≥2 heads. No learning happened.

---

## 1. Current Assumptions (What We're Betting On)

run11 has three big architectural bets:

| Bet | Rationale | Risk |
|-----|-----------|------|
| P5 features + 6-layer T → PSR learns | Loss dropped 10× | Event F1 may still be 0 if predictions are flat |
| TAL topk=10 + P3/P4/P5 → detection learns | Denser supervision, better features | May need more epochs / data augmentation |
| 3-layer MLP → activity learns from cls_token | More capacity than 2-layer | cls_token may be fundamentally overloaded for 75 classes in MTL |

**The null hypothesis we're testing:** The EP10 0.0/0.58%/0.004 was a *head architecture* problem, not a *MTL gradient competition* problem. If run11 EP10 shows meaningful signal on all heads, the null hypothesis holds — MTL works, the heads just needed better features/capacity.

---

## 2. Path A: Run run11 to EP30, Evaluate, Write Paper

**What:** Let run11 train through ep30 (~18 hours from now). Collect ep10, ep20, ep30 evals. Write the paper based on those numbers.

**When run11 shows:** Activity >5%, Detection >0.05, PSR >0.1 at ep10 — ascending at ep20/ep30.

**Time required:** 2 days (training + eval + paper write)

**Paper story:** "One MViTv2-S backbone, four specialized heads, Kendall-collapse diagnosed and fixed. MTL provides bounded-cost sharing on activity/detection, positive transfer on pose, parameter efficiency of ~3× vs matched specialists."

**Advantages:**
- Fastest path to a paper
- Uses code we already have running
- Efficiency claim is genuine (one backbone forward pass for 4 tasks)
- Kendall-collapse story is novel and defensible
- PSR improvement (1.56→0.17 loss) is a strong methodological result

**Risks:**
- EP10 numbers may still be low (same as run10). Then we need Plan B.
- Without ST baselines, can't compute MTL/ST ratios → paper is weaker
- Activity at 5-15% is still far from 65% SOTA → need to frame as "MTL cost"
- Reviewer may ask: "Why not just train separate models?"

**Expected outcome quality:** Good to strong paper (L2+L3+method), dependent on ep10 numbers.

---

## 3. Path B: Complete run11 + Run 4 ST Baselines → Compute MTL/ST Ratios

**What:** Let run11 train through ep30. Simultaneously launch single-task training on GPU 2 for all 4 tasks. Compare MTL vs ST at matching epochs. Write paper with quantitative MTL/ST evidence.

**Time required:** 5-7 days (run11 training 2 days + ST baselines 3-4 days on GPU 2 + paper write 1 day)

**Required hardware:** GPU 1 (run11 MTL) + GPU 2 (ST baselines, sequential or staggered)

**ST baseline scripts:** `scripts/train_st.py --task {det,act,psr,pose}` — already implemented, just needs launching.

**Paper story:** Same as Path A, but with MTL/ST ratio as the headline quantitative result: "On pose, MTL outperforms single-task by X. On activity, MTL achieves Y% of the single-task baseline at 3× parameter efficiency. Across all 4 tasks, the shared backbone captures Z% of specialists' performance."

**Advantages:**
- Quantitative evidence for "MTL helps" (ratio > 1) or "MTL doesn't hurt" (ratio > 0.8)
- ST baselines are mandatory for any serious paper (Opus 192 §5, Opus 181 §4, Opus 186 §2)
- If MTL pose > ST pose, that's a publishable finding
- Model soup option becomes available (average ST backbones → warm MTL init)
- Reviewer can't ask for ST baselines — they're already done

**Risks:**
- ST baselines may be much better than MTL → ratio < 0.5 on some heads → harder story
- GPU 2 has 16GB — ST detection training may need batch-size tuning
- Sequential baselines take 3-4 days even with GPU 2
- PSR 70.9M head doesn't have an ST equivalent (need to decide: same head ST or smaller?)

**Expected outcome quality:** Strong paper with quantitative evidence.

---

## 4. Path C: All of Path B + Model Soup + Soup-Finetune

**What:** Path B, then average the 4 ST backbone weights into a "soup," initialize MTL from soup, finetune 10-20 epochs. Report the soup-finetune result as the main MTL number.

**Time required:** 7-10 days (Path B 5-7 days + soup 1 hour + soup-finetune 2-3 days)

**Paper story:** "We first train 4 single-task specialists on a shared MViTv2-S architecture. Averaging their backbone weights produces a warm initialization for MTL. After fine-tuning, the souped MTL model matches or exceeds the individual specialists on 3 of 4 tasks while using 3× fewer parameters."

**Advantages:**
- Soup init is a near-free improvement (Opus 192 Q5: "nice-to-have init, not a dependency")
- May close the MTL/ST gap on the struggling heads
- Strong ablation: "MTL from scratch vs MTL from soup"
- Appealing to reviewers: "We show that even simple weight averaging provides useful MTL initialization"

**Risks:**
- Soup averaging across 4 different tasks may land between basins (worse than any ST) — per Opus 192 Q5 caveat
- Adds 2-3 days to timeline
- Soup success depends on ST baselines all using the same architecture → need to verify ST scripts use same backbone config

**Expected outcome quality:** Strongest possible paper from this architecture class.

---

## 5. Path D: Abandon MViTv2-S → Scale or Switch Backbone

**What:** If run11 EP10 still shows zero on 2+ heads, the conclusion is: MViTv2-S, even with good heads, cannot support 4 tasks simultaneously. Switch to a larger backbone or a fundamentally different architecture.

**Options:**
- **D1: MViTv2-L (53M, CC-BY-NC):** 2× larger backbone, same architecture, mild hedge (Opus 192 §4: "a mild, license-clean hedge"). Cost: 5-7 days retraining.
- **D2: InternVideo2-L (304M, license risk):** Foundation video model. Opus 192's "single worst idea" — inverts efficiency claim, breaks WACV comparison, license unclear. Only as headroom ablation.
- **D3: DINOv2-L (304M):** Foundation image model, Apache license. No temporal modeling built in. For detection/pose only.

**Time required:** 10-14 days minimum

**Paper story:** "We scale the backbone to show the MTL ceiling. At 53M parameters, the shared model reaches X% of specialists. The 34.5M model captures Y% of the scaled model's performance at Z% of the parameters."

**Advantages:**
- Shows what's possible with more capacity
- If MViTv2-S genuinely can't handle 4 tasks, scales the claim

**Risks:**
- Opus 192 explicitly rejected this path: "capacity isn't the bottleneck"
- "10-14" days is optimistic — training instability, debugging, new architecture bugs
- May end up with the same problems at 2× compute cost
- If the problem is MTL gradient competition, not capacity, larger backbone doesn't fix it

**Expected outcome quality:** Unclear. High risk of dead end per Opus 192's analysis.

---

## 6. Path E: Hybrid — Per-Head Triage

**What:** Based on EP10 results, triage each head:

| If EP10 shows... | Action |
|-----------------|--------|
| Pose ≤9° MAE and improving | ✅ No changes. Positive transfer candidate. |
| PSR F1 > 0.05 and loss dropping | ✅ Continue. The P5 fix is working. |
| PSR F1 < 0.02 despite low loss | ⚠️ Loss is low but predictions are flat (all "no transition"). Need threshold tuning or asymmetric loss. |
| Activity > 5% top-1 | ✅ Continue. 3-layer MLP is learning. |
| Activity < 3% top-1 | ⚠️ cls_token is overloaded. Consider: VideoMAE stream, separate temporal branch, or activity-specific log_var uncap. |
| Detection > 0.05 mAP | ✅ Continue. TAL + P5/P4/P3 is working. |
| Detection < 0.01 mAP | ⚠️ Run Probe 1 (overfit-200) to isolate eval vs feature vs assigner bug. If eval works on overfit, the problem is features. If eval fails, fix eval first. |

**Advantages:** Surgical — only fix what's broken. Saves time vs full paths.

**Risks:** May lead to one-head-at-a-time whack-a-mole. No guarantee the fixes compose.

---

## 7. Decision Matrix

| Path | Time (days) | Risk of Dead End | Paper Quality (if works) | Requires ST Baselines | Opus 192-Compliant |
|------|------------|-----------------|------------------------|----------------------|-------------------|
| **A**: run11 → ep30 → paper | 2 | Medium | Good | ❌ | ⚠️ Partial (missing ST) |
| **B**: A + ST baselines | 5-7 | Low | Strong | ✅ | ✅ |
| **C**: B + Soup + finetune | 7-10 | Low-Medium | Strongest | ✅ | ✅ |
| **D**: Scale backbone | 10-14 | High | Unclear | Depends | ❌ (explicitly rejected) |
| **E**: Per-head triage | 3-5 | Medium | Variable | ❌ (best with ST) | ⚠️ Partial |

---

## 8. The Real Strategic Question

**The pre-committed bet (run11) assumes the old heads were too small / reading wrong features — not that MTL fundamentally can't work. If EP10 validates this (PSR loss drop is already strong evidence), then:**

1. **Path B is the minimum publishable paper.** MTL/ST ratios + Kendall-collapse + efficiency numbers.
2. **Path C is the strongest paper** from this architecture. Soup init as a clean ablation.
3. **The paper's thesis is:** "We show that a single MViTv2-S backbone, with properly-sized heads and fixed Kendall weighting, can serve 4 assembly-understanding tasks at bounded cost vs specialists, with positive transfer on pose. The Kendall-collapse diagnosis is the methodological contribution."

**If EP10 still shows 0.0 on 2+ heads, then MViTv2-S genuinely can't support 4 tasks and we need to either:**
- Accept a weaker paper (MTL doesn't work, here's why — the Kendall pathology paper)
- Drop to 3 tasks (pose + PSR + activity, cut detection)
- Scale backbone (Path D, against Opus 192 advice)

---

## 9. Recommendation to Opus

**We ask Opus to decide between Path A, B, C, or a hybrid, based on the evidence in files 195-198.** The specific prompt is in file 200.

Key facts Opus needs:
1. PSR loss dropped 10× (1.56→0.17) from feature source fix alone — validates diagnosis
2. Detection alternates 0.001/4-5 loss — TAL is providing real gradient signal
3. Activity loss is still high (4-5) despite 3-layer MLP — cls_token may be overloaded
4. Pose is healthy (~9° MAE) — strongest positive transfer candidate
5. ST baselines are mandatory (per all 3 Opus rounds) but haven't been run
6. GPU 2 (16GB) is available for ST training
7. run11 is live and will provide EP10 signal in ~6 hours
