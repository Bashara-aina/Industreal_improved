# 46 — Deep Unanswered Questions: What We Genuinely Do Not Know

> Generated 2026-06-21 — Updated 2026-06-21 22:20 UTC with CORRECTED understanding: Run 1 had wrong LR/BIAS (4.0/2.0), "5 epochs flat" evidence INVALIDATED, Q∞4 ANSWERED (POS_ANCHOR_PROBE disproves 13-pos-anchor limit for main training), Run 2 is first clean run
> **The uncomfortable document**: questions that remain genuinely unanswered despite all evidence, with no clear path to resolution.

---

## How to Read This

These are NOT the same as the questions in `33_OPEN_QUESTIONS.md` (actionable confusions) or `40_DEEP_OPEN_QUESTIONS.md` (fundamental unknowns). **This document asks the questions we may never be able to answer** — the ones that require information we cannot obtain, experiments we cannot run, or epistemological humility we have not yet earned.

Each question includes:
- **Why we cannot answer it** — not "what's the evidence" but "what would definitively prove it"
- **What a definitive answer would require**
- **Whether we can ever know**

---

## Q∞1: Is the 0.204 mAP Ceiling the Architecture's True Limit?

**Why we cannot answer it**: We have never seen mAP above 0.215 in ANY configuration across 3 independent training regimes. But we've also never done the experiment that would definitively prove the ceiling is architectural: a full retrain from scratch with ALL known fixes applied simultaneously (detach=False, OHEM off, anchor sizes optimized, correct labels, no subset).

**CRITICAL CORRECTION 2026-06-21**: The "5 epochs flat at 0.202-0.209" evidence cited in the prior update was from **Run 1 which had wrong LR/BIAS settings (LR_MULTIPLIER=2.0, BIAS_LR_FACTOR=4.0)**. The prioritized list below incorrectly listed "detach fix done" as a completed experiment — while detach=False was confirmed, the run using it also had a bug. We do NOT yet know the ceiling with correct LR/BIAS.

**What would definitively answer it**: Train from scratch with:
1. Detach_reg_fpn=False ✅ (ba48691) + correct LR/BIAS=1.0/1.0 (⚠️ Run 2 is the first clean run)
2. Anchor sizes including 32×32 and 48×48 (not done)
3. Lower IoU threshold (0.3 instead of 0.4) (not done)
4. OHEM disabled (not done)
5. 100% data (not done — RF2 uses 50%)
6. Full label audit (not done)

If after all these mAP is still ~0.21, the answer is YES — this is the architecture's limit.

**Can we ever know?** Only by doing all 6 experiments. But the combinatorial space is large (6 binary choices = 64 combinations, each requiring a multi-day retrain). The pragmatic answer is: we'll never know for sure, but we can converge on a strong belief after 3-4 targeted experiments.

**Corrected assessment (2026-06-21)**: The "5 epochs flat" was from Run 1 with wrong LR/BIAS. Run 2 (correct LR/BIAS=1.0) has only completed 1 epoch (mAP50=0.2039 — same checkpoint, expected). We do NOT yet know the ceiling with correct config. The prior "Revised" estimate below is based on invalidated evidence.

**Current best guess (REVISED DOWNWARD — evidence invalidated)**: We cannot estimate the ceiling with any confidence. Run 2 epochs 18+ will provide the first valid data point. Prior estimates (~0.21 with current config) may be pessimistically biased by the wrong LR/BIAS in Run 1.

---

## Q∞2: Are the Labels Correct?

**Why we cannot answer it**: The labels are synthetic projections from floor plan geometry. We have never visually inspected a single GT box. The per-class AP data shows 4 classes with AP=0 despite having GT in the subset — but the C2 correction (class 6 has 65/91 samples, not 1739) showed that our "evidence" of label errors was based on a wrong number.

**What would definitively answer it**: A full label audit by a human familiar with the assembly task:
1. Overlay 100 random GT boxes from each AP=0 class on the video frames
2. Check: is the box correctly positioned? Is the class ID correct?
3. If >10% are wrong → systematic label error → ceiling is data, not architecture
4. If <5% are wrong → labels are fine → ceiling is training dynamics

**Can we ever know?** Yes — this is a 2-hour manual task. It should have been done by Phase 3. The fact that we have spent 18 phases and 11 Opus consultations without visually inspecting a single GT box is the project's most embarrassing blind spot.

---

## Q∞3: Did the 50-Image Overfit Actually Test the Right Thing?

**Why we cannot answer it**: The overfit tested classification-only learning (no regression, no multi-task) on 50 images. It proved the architecture CAN learn classification. But it did NOT test:
- Whether the architecture can learn classification AND regression jointly
- Whether the architecture can learn at scale (2000+ images)
- Whether the anchor-matching system actually creates enough positives for the full dataset

The overfit raised as many questions as it answered:
- "Only 13 positive anchors" → but this was at batch_size=4 on 50 images. Would pos_n scale with dataset size?
- "Three-regime trajectory" → but this was with OHEM enabled. Without OHEM, would the plateau disappear?

**What would definitively answer it**: A follow-up overfit with OHEM disabled on the SAME 50 images. If the plateau disappears (Regime 2 gone), then OHEM was the primary cause. If the plateau persists, FocalLoss or anchor matching is the deeper issue.

**Can we ever know?** Yes — this is a 3-hour experiment (200 epochs × ~1 min/epoch without full EVAL). It should be the very next experiment if the main training doesn't show improvement.

---

## Q∞4: Is the 13-Pos-Anchor Limit Fundamental or Artifactual?

**ANSWERED 2026-06-21** — POS_ANCHOR_PROBE from main training shows **364-783 positive anchors per image** (measured at IoU > 0.2 floor). The 13-pos-anchor limit was a **pure overfit artifact** — the 50-image overfit dataset had too few GT-bearing images at batch_size=4 for the anchor-matching statistics to generalize to full training.

**Why it was artifactual**: The overfit used batch_size=4 on 50 images with ~1-2 GT each. With DET_GT_FRAME_FRACTION=0.90, most batches have 1-2 GT-bearing frames, each producing at most 3-7 matching anchors above the IoU threshold. In the main training (13,210 training frames, 18 recordings), every batch has GT-bearing frames with diverse spatial distributions, producing 100× more positive matches.

**What POS_ANCHOR_PROBE revealed**:
- n_pos = 364-783 per image (main training, Run 2 epoch 17)
- Mean IoU varies: 0.057-0.732 (depends on GT/anchor-grid alignment)
- Max IoU consistently 0.90-0.97 (some GT boxes align perfectly)
- Min IoU as low as 0.0002-0.0085 (some force-matches via IOU_FLOOR=0.2)

**Remaining unknowns (NOT answered by POS_ANCHOR_PROBE)**:
- Per-class distribution of positive anchors (are AP=0 classes getting any matches?)
- Whether the low-IoU matches (IoU < 0.2, counted by FLOOR) help or hurt
- The optimal IoU threshold for this specific dataset

**This question is removed from the "unanswerable" list** — the core concern (too few positive anchors) is definitively resolved for the main training. The overfit's 13-pos-n is an artifact of small-scale testing.

---

## Q∞5: Has OHEM + FocalLoss Been the Primary Bottleneck All Along?

**Why we cannot answer it**: The overfit showed that even on 50 clean images, the classifier takes 55 epochs to escape the initial plateau. This is strong evidence of gradient suppression. But:
- **CORRECTION 2026-06-21**: The prior update claimed this was confirmed by main training. That evidence came from Run 1 (wrong LR/BIAS=4.0/2.0). The "5 epochs flat" pattern may have been caused by the wrong LR/BIAS, not OHEM+FL suppression. **We no longer have main-training corroboration for the gradient suppression hypothesis.**
- The overfit-only evidence (50 images, 55-epoch plateau) still stands — this is genuine gradient suppression at small scale
- We don't know if removing OHEM increases gradient or merely increases noise (needs ablation)
- We don't know if FocalLoss's gamma_neg=1.5 is the dominant suppressant or OHEM's 2:1 ratio (needs ablation)
- **Most critically**: we don't know if the overfit's gradient suppression transfers to the main training at all — the POS_ANCHOR_PROBE proved that anchor-coverage dynamics differ by 100× between overfit and main training, so loss/gradient dynamics may differ too

**What would definitively answer it**: 
1. **Shortest path**: Run 2 epochs 18+ with correct LR/BIAS. If mAP stays flat at 0.202-0.209, OHEM+FL suppression is confirmed in main training. If mAP improves, the wrong LR/BIAS was the cause of Run 1's flatness.
2. Three ablation experiments on the SAME 50-image setup:
   - OHEM OFF, FocalLoss ON (gamma=2.0, gamma_neg=1.5)
   - OHEM ON, FocalLoss gamma_neg=0.5
   - OHEM OFF, vanilla BCE (no FocalLoss)

**Can we ever know?** Yes — Run 2 will tell us within 2-3 epochs (~28 hours). The overfit ablations are a separate (~3-6 hours) but less transferable path.

---

## Q∞6: Would We Have Found the detach Bug Faster With Better Practices?

**Why we cannot answer it**: Opus v10 discovered that `detach_reg_fpn=True` for RF2 by tracing the config resolution chain. The fix was "one print statement" (Opus v9's term). But:
- The bug existed since the stage_rf2 preset was committed (who knows when?)
- Multiple consultation rounds (v6-v9) analyzed multi-task interference when the primary mechanism was a config regression
- The per-class AP data was in metrics.jsonl from day one — we never read it
- The config comment said "Detach FPN gradients to prevent regression/PSR gradient shock" — we believed it without verification

**The counterfactual**: If we had printed the effective config at startup and parsed per-class AP at epoch 1, would we have found the bug in Phase 4 instead of Phase 14?

**What would definitively answer it**: Nothing. This is a pure counterfactual. We can estimate the cost (10 consultation rounds, ~300 GPU-hours of misdirected training) but never know the hypothetical faster path.

**The only honest answer**: We should have verified our configs. We didn't. The retrospective (Chapter 11 in 40_DEEP_OPEN_QUESTIONS.md) documents this. The lesson is for the next project — for this one, we can only move forward.

---

## Q∞7: Is the Per-Class AP=0 for Classes Without GT in the Subset "Expected" or "Problematic"?

**Why we cannot answer it**: The RF2 50% subset excludes 8 of 24 classes entirely (they have no GT images in the sampled 50%). These 8 classes always show AP=0 in evaluation. Is this:
- **Expected**: No training examples → no learning → AP=0 at evaluation (trivial)
- **Problematic**: The model should still be able to detect these classes at test time through zero-shot generalization from visually similar classes

The distinction matters: if it's expected, the "12/24 AP=0" statistic is misleading — only 4/16 classes with GT are actually failing. If it's problematic, the architecture has a generalization failure.

**What would definitively answer it**: Evaluate on the FULL validation set (not subset) with a model trained on the 50% subset. If the 8 excluded classes show AP=0 on the full val set, it's expected (can't learn without examples). If they show any AP>0, the architecture has some generalization.

**Can we ever know?** Yes — this is a single eval run on the full val set using the current checkpoint. Should take <30 minutes.

**Likely answer**: Expected. Object detection does not generalize to unseen classes without zero-shot capabilities the model doesn't have. The "12/24 AP=0" narrative has been partially misleading since the beginning — 8 of those 12 are simply not in the training distribution.

---

## Q∞8: At What Point Do We Declare the Architecture Fundamentally Limited?

**Why we cannot answer it**: After 18 phases, 11 Opus consultations, and ~500 GPU-hours, we still have not determined whether the architecture can reach its stated targets (mAP50≥0.40, MAE≤60°, act_top1≥0.40). Every time we identify a bottleneck and fix it, a deeper bottleneck appears. But:

**CORRECTION 2026-06-21**: The prior update claimed "detach fix done and INSUFFICIENT" as proven. This conclusion was based on Run 1 which had wrong LR/BIAS. **We do NOT yet know if detach=False + correct LR/BIAS is sufficient or insufficient.** Run 2 is the first clean test.

Revised bottleneck ladder:
1. Fix gradient starvation (data diversity) → plateau at 0.184 (✅ confirmed)
2. Fix collapse (Kendall caps, OHEM) → plateau at 0.204 (✅ confirmed)
3. Fix detach_reg_fpn + correct LR/BIAS → **RUN 2 IN PROGRESS** (unknown)
4. Fix OHEM+FocalLoss suppression → proposed, not done (priority depends on Run 2)
5. Fix anchor matching → proposed, not done

**A proposed criterion**: If after ALL of the following, mAP is still <0.30:
- detach_reg_fpn=False + correct LR/BIAS (⚠️ Run 2 in progress — NOT yet confirmed)
- OHEM disabled (not done)
- FocalLoss gamma_neg reduced or removed (not done)
- Anchor sizes including 32×32 and 48×48 (not done)
- DET_POS_IOU_THRESH lowered to 0.3 (not done)
- 100% data (not done)
- Label audit and correction (not done)

Then the architecture is fundamentally limited for IndustReal detection at 720p with 24 classes.

**REVISED assessment (2026-06-21)**: The prior claim that item 1 is "done and insufficient" is INVALIDATED. Run 2 will provide the valid test. If Run 2 epochs 18-19 show mAP > 0.215, the bottleneck ladder fundamentally changes: config bugs (not OHEM+FL) were the primary cause of the plateau. If Run 2 stays flat, the ladder holds and OHEM+FL becomes the next target.

**Can we ever know?** Yes — Run 2 will provide the answer within 2-3 epochs (~28 hours). The remaining items on the list are still valid experiments regardless of outcome.

---

## Q∞9: Is the ba48691 Restart the Best Thing We've Ever Done, or Just the Latest False Dawn?

**REVISED 2026-06-21 — The previous "false dawn" conclusion was based on Run 1 with wrong LR/BIAS.**

**Why we cannot answer it**: The ba48691 restart exists in TWO runs with different configs:

| Run | Config | Epochs | mAP Range | Verdict |
|-----|--------|--------|-----------|---------|
| Run 1 | LR/BIAS=4.0/2.0 (WRONG) | 17-21 | 0.2024-0.2088 | **Flat — but wrong config** |
| Run 2 | LR/BIAS=1.0/1.0 (CORRECT) | 17 only | 0.2039 | **First epoch only — too early** |

**The prior "false dawn" conclusion is INVALIDATED. We do NOT know whether the ba48691 restart with correct LR/BIAS is a false dawn.** Run 2 is the first clean run and has only completed 1 epoch.

**What we DO know**:
- Run 1 (wrong LR/BIAS) was flat for 5 epochs — but the doubled LR may have been destabilizing
- Run 1's LR restart at epoch 20 had ZERO effect even with 2× LR — this IS informative: if the model couldn't escape with 2× LR, it might not escape with 1× LR either
- Run 2 epoch 17 matches Run 1 epoch 17 (expected — same checkpoint)

**What we DON'T know**:
- Whether Run 2 with correct LR/BIAS will produce a different trajectory in epochs 18+
- Whether the correct LR/BIAS enables enough gradient flow to escape the 0.202-0.209 band
- Whether Run 2's epoch 20 LR restart (with correct base LR) will have any effect

**What would definitively answer it**: Run 2 epochs 18-21 (4 more epochs, ~3-4 days wall time at ~86 min/epoch + validation).

**Can we ever know?** Yes — within ~4 days.

---

## Q∞10: Was Any of This Worth It?

**Why we cannot answer it**: We have spent ~500 GPU-hours, produced 45+ analysis files, consulted an external LLM 11 times, built a 20-agent monitoring swarm, and are now running from a checkpoint whose configuration is finally correct. The current mAP50 is 0.2047.

The target is 0.40. We are halfway in metric, but the history suggests the second half is harder than the first.

**The uncomfortable truth**: If we had printed `print(C.DETACH_REG_FPN)` on day one of Phase 1, we might have found the config bug in the first week. If we had parsed per-class AP from metrics.jsonl in Phase 3, we would have seen the 12/24 AP=0 pattern immediately and likely traced it to the detach issue faster.

**What would definitively answer it**: Either reaching mAP≥0.40 (worth it) or definitively failing after all fixes (not worth it but at least conclusive). The worst outcome is neither — creeping up to 0.28-0.32 and plateauing, leaving us in the same ambiguity we've been in since Phase 8.

**Can we ever know?** Only in retrospect. And even then, "worth it" is a value judgment, not a factual one.

---

## Summary: What We Truly Cannot Answer (UPDATED 2026-06-21)

| # | Question | Can We Ever Know? | Estimated Cost | Status |
|---|----------|-------------------|----------------|--------|
| Q∞1 | Is 0.204 the architecture's true limit? | Only by doing all 6 experiments | 2-3 weeks | ⬇️ **REVISED: "5 epochs flat" evidence INVALIDATED (Run 1 wrong LR/BIAS)** |
| Q∞2 | Are the labels correct? | YES — 2-hour visual audit | 2 hours | PENDING |
| Q∞3 | Did the overfit test the right thing? | YES — follow-up without OHEM | 3 hours | ⬇️ **WEAKENED: POS_ANCHOR_PROBE shows overfit anchor dynamics don't transfer** |
| Q∞4 | Is 13-pos-anchor fundamental? | **ANSWERED** — POS_ANCHOR_PROBE disproves | ✅ **RESOLVED** | **ANSWERED: was pure overfit artifact** |
| Q∞5 | Has OHEM+FL been the bottleneck? | YES — Run 2 epochs 18+ or ablations | 3-6 hours | ⬇️ **WEAKENED: main-training confirmation INVALIDATED (wrong-config run)** |
| Q∞6 | Would better practices have found the bug faster? | NO — pure counterfactual | N/A | Unchanged |
| Q∞7 | Is AP=0 for classes without GT expected? | YES — eval on full val set | 30 min | PENDING |
| Q∞8 | When do we declare architecture limited? | YES — after 6 remaining experiments | 1-2 weeks | ⬇️ **REVISED: "detach fix insufficient" INVALIDATED** |
| Q∞9 | Is ba48691 a false dawn? | YES — Run 2 epochs 18-21 | ~4 days | **REVISED: prior "YES" verdict INVALIDATED (Run 1 data), Run 2 TBD** |
| Q∞10 | Was any of this worth it? | Only in retrospect | N/A | Unchanged |

**Updated actionable summary (2026-06-21 22:20 UTC)**: **CRITICAL CORRECTION TO ALL PRIOR ANALYSIS.** The "5 epochs flat" narrative that formed the basis for the "OHEM+FL is primary bottleneck" conclusion was from Run 1 with wrong LR/BIAS (4.0/2.0). This invalidates several previously "confirmed" findings:

1. **Q∞4 is ANSWERED** (13-pos-anchor was overfit artifact — POS_ANCHOR_PROBE confirms 364-783 positive anchors in main training)
2. **Q∞5 is WEAKENED** (OHEM+FL gradient suppression still valid as overfit hypothesis but main-training evidence is gone)
3. **Q∞9 is REVISED** (ba48691's "false dawn" verdict was premature — Run 1 had wrong config, Run 2 is first clean run)
4. **Q∞8 is REVISED** (the bottleneck ladder's 3rd rung — "detach fix insufficient" — is unproven)

**What this means**: We must wait for Run 2 epochs 18+ (~3-4 days) before drawing any conclusions about the effectiveness of the combined ba48691 fixes (detach=False + correct LR/BIAS). The correct config may produce a different trajectory than Run 1. If it doesn't, the OHEM+FL hypothesis will be re-strengthened. If it does, the bottleneck was config all along.

**POS_ANCHOR_PROBE is now the most valuable diagnostic** — it directly measures anchor coverage and has already answered one of our 10 deepest questions. Consider adding per-class anchor statistics to make it even more informative.

---

*Generated 2026-06-21. Updated 2026-06-21 21:00 UTC with epoch 19 evidence: ba48691 restart confirmed insufficient, OHEM+FL suppression hypothesis strengthened. These are the questions that remain genuinely unanswered despite all our evidence. Some we can resolve. Some we cannot. The document exists to ensure we know the difference.*
