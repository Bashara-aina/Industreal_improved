# 191 — Opus Round 3 Prompt: Validate the Plan to Make MTL Beat SOTA

**Date:** 2026-07-09
**Purpose:** Single-prompt consultation to Opus for round 3. Explains the current state, the proposed plan (files 188-190), and the strategic questions we need answered.
**Audience:** Opus (the model being consulted). Should be copy-pasted into a fresh Opus session as the opening prompt.
**Companion files (read in this order):** 187 → 188 → 189 → 190

---

# OPENING PROMPT — for Opus round 3

Hello Opus. I'm consulting you on a multi-task learning (MTL) research project where we've made progress and now need strategic validation before committing 2-3 more weeks of compute. Please read the following context and answer the questions at the end.

## The Project

We're training a **4-task MTL model on the IndustReal assembly dataset** (Schoonbeek et al., WACV 2024, arXiv:2310.17323 — please verify against the paper directly as you did in round 2). The four tasks are:

- **Detection** (24 classes including background, MViTv2-S-style FPN head)
- **Activity** (75-class action recognition on the MViT class token)
- **PSR** (per-frame state transitions for 11 assembly components)
- **Pose** (head 6D forward/up vector regression)

The goal is to publish at AAIML (deadline ~3-4 weeks out). Our hypothesis: **MTL with one shared backbone is more efficient, faster, and at least as accurate as single-task specialists on each head, often beating SOTA.**

## What's Already Done (file 187 = `187_OPUS_181_186_IMPLEMENTATION_CHECKLIST.md`)

In rounds 1 and 2 (your answers in files 181 and 186), you identified three factual corrections and a Path-D implementation. We've implemented ALL of the following (verified by smoke test, training live on PID 2545563):

### Opus 181 (Path D — round 1)
- **D1:** Per-task EMA loss tracker; each task's loss divided by its own EMA before the Kendall term
- **D1b:** Sqrt-tame activity class weights (137 → ~12 max ratio)
- **D1c:** Activity label_smoothing 0.1 → 0.05
- **D2:** Per-task log_var caps: det≤4.0, act≤1.0, psr≤0.5, pose≤4.0
- **D3:** Keep Kendall + PCGrad
- **D4:** Move `optimizer.zero_grad()` from top of train_step to after `step()` (initial)
- **D4-fix:** PCGrad backbone override ACCUMULATES (not overwrites)
- **Detection:** 3×3 positive cells (pos_radius=1) + focal α=0.5

### Opus 186 (round 2 corrections)
- **§5.1:** Grad-accumulation now divides total_loss by `grad_accum_steps` (was summing, causing 2× effective LR + over-clipping)
- **§5.2:** Per-cell DFL/IoU targets in the 3×3 patch (was using GT center for all 9 cells)
- **B-6:** PSR feature source switched from `fpn_feats["P2"]` (conv_proj, 96ch, raw patch embeddings) to `fpn_feats["P5"]` (blocks[14], 768ch, semantic features)
- **E-3:** EMA model weights (momentum 0.999) swapped at evaluation
- **E-6:** Grad-clip 1.0 → 5.0 (configurable via `--grad-clip-norm`)
- **E-7:** `max_batches_per_epoch` default 0 → 8000 (2× data coverage)
- **Q3:** 2-layer MLP for activity head (insurance)
- **Resume:** Pre-filter state_dict by shape (PSR head reshape 96→768ch, activity 1→2 layers)

## Current Training State (Opus 186 also reframed the bar)

After 7 epochs of the Path-D run with all fixes applied, the picture is:

| Head | Current (ep7) | SOTA (WACV paper) | 80% bar | Status |
|------|---------------|-------------------|---------|--------|
| Detection | 0.0 mAP | 0.838 | 0.67 | Per-cell DFL fix working; loss variable batch-to-batch |
| Activity | 0.008 top-1 | 0.652 | 0.52 | 2-layer MLP is fresh-init; loss flat at 4.83 (needs 5-10 epochs) |
| PSR | 0.0 F1 | 0.901 | 0.72 | 768ch head is fresh-init; loss flat at 1.58 (uncertain) |
| Pose | 10° fwd MAE | No SOTA (WACV paper) | n/a | Healthy already |

**Honest assessment:** Path-D alone reaches ~50-60% of SOTA per head by ep30-100. **It does NOT clear the 80% bar across all heads.** The remaining gap is **representational + capacity**, not optimization.

## The Proposed Plan (files 188-190)

To support the hypothesis that MTL is helping (not hurting), I've drafted three deep documents proposing architectural changes. **Please review these files and validate / critique the plan:**

### File 188: Per-Head Architecture Upgrades (`188_PER_HEAD_ARCHITECTURE_UPGRADES.md`)

**Per-head redesigns:**
- **Detection:** Replace hand-rolled head with **YOLOv8 head** (decoupled cls/reg branches + TaskAlignedAssigner + DFL). Re-implement from the YOLOv8 paper (not copy Ultralytics code) to avoid AGPL-3.0 license issues.
- **Activity:** Replace 1-layer / 2-layer MLP with **temporal attention pooling** (learnable query attends to per-frame tokens) + **ArcFace loss** (additive angular margin for long-tail).
- **PSR:** Replace 3-layer transformer on P5 features with **STORM-like state-transition decoder** (2-layer GRU + per-component transition head).
- **Pose:** Keep current (already healthy; no SOTA exists).
- **Backbone plumbing:** Expose per-frame tokens (not just pooled class token) so activity head can do attention pooling.

**Cost:** 8-12 days engineering.
**Expected per-head:** 0.55-0.80 detection, 0.30-0.50 activity, 0.50-0.80 PSR, 4-6° pose.
**Confidence:** ~60% to clear 80% SOTA bar across all heads.

### File 189: Backbone & MTL Topology (`189_BACKBONE_AND_MTL_TOPOLOGY.md`)

**Backbone options:**
- **MViTv2-L** (53M, 1.5× current; cheap, low upside)
- **InternVideo2-L** (304M, frozen + LoRA; strongest features BUT license caveat per your round 2 §5.3)
- **DINOv2-L** (305M, image-only, Apache 2.0; safer license)
- **EVA-02-L** (305M, MIT license; best image classifier)

**MTL topology options:**
- Shared backbone (current)
- LoRA adapters per task
- MMoE (mixture of experts)
- Cross-task attention (high risk, Opus 186 §2 Q7 said skip)
- **Sequential pretraining + model soup (Strat-4)** — most defensible, highest cost

**Three tiers proposed:**
- **Tier 1 (cheap, ~10 days):** MViTv2-L + per-head upgrades + shared MTL → 70% SOTA
- **Tier 2 (recommended, ~15 days):** Frozen foundation + LoRA + per-head upgrades → 80% SOTA
- **Tier 3 (full, ~25 days):** Tier 2 + sequential pretrain + model soup → 95% SOTA

### File 190: Training Path & Hypothesis Validation (`190_TRAINING_PATH_AND_HYPOTHESIS_VALIDATION.md`)

**3-phase plan:**
- **Phase 1 (3-5 days):** Per-head architecture implementation + smoke tests
- **Phase 2 (4-5 GPU-days):** 4 single-task pretrainings on the chosen backbone (the comparison baseline)
- **Phase 3 (5-7 GPU-days):** MTL finetune from soup initialization (Wortsman 2022 / Ilharco 2022)

**Hypothesis validation framework:**
- Per-head ST vs MTL comparison (MTL/ST ratio target ≥0.9 = "MTL is helping")
- Efficiency claims: 1/9th the parameters of 4 specialists
- Training efficiency: 3-4× faster wall-clock
- If MTL < ST on some head: accept L2+L3+method story (per your round 1 §0)

## Specific Questions for You (Opus)

Please answer each of these — they're decision-changing.

### Strategic (file 188 / 189 framing)

**Q1.** Is the **YOLOv8 head re-implementation** the right call for detection? Or should we keep the current hand-rolled head and invest the engineering time elsewhere? Specifically: do you agree that **YOLOv8's TAL+decoupled head is the load-bearing change** for closing the 0.0 → 0.67 mAP gap? If not, what's the alternative?

**Q2.** Is the **STORM-like decoder for PSR** the right architecture, or is there a simpler approach? The current 3-layer transformer on P5 features is flat at loss 1.58. Is the STORM decoder (2-layer GRU + per-component transition head) likely to break the loss plateau, or is the bottleneck somewhere else (data, features, loss formulation)?

**Q3.** Is the **temporal attention pool + ArcFace** the right activity upgrade? The 2-layer MLP is fresh-init and may converge on its own. ArcFace is well-validated for face recognition but unproven on 75-class assembly activity. Should we try it, or skip and trust the 2-layer MLP?

**Q4.** Should we use **frozen foundation backbone (InternVideo2-L or DINOv2-L)** or **scale up MViTv2 (S→L)?** Specifically: given the 304M foundation models are 6× larger than MViTv2-S, is the inference latency / VRAM cost worth the expected 15-25% per-head gain? Or is MViTv2-L (1.5× scale, 53M) the right hedge?

**Q5.** Is the **sequential pretraining + model soup (Strat-4)** plan worth the 2-3 week wall-clock? Or should we trust Tier 2 (frozen foundation + LoRA + per-head upgrades) to give us 80% SOTA without the soup overhead?

### Hypothesis validation (file 190 framing)

**Q6.** Is **MTL/ST ratio ≥0.9 across all 4 heads** the right threshold for "MTL is helping"? Or should we accept a lower threshold (≥0.7) and focus on the efficiency story instead?

**Q7.** If MTL significantly underperforms ST on one head (e.g., PSR reaches 0.85× of ST), do we have a defensible paper? Or does a single soft head break the hypothesis entirely?

**Q8.** Is the **headline table** in file 190 §5.3 (MTL 0.96× ST on detection, 0.97× on activity, 0.97× on PSR, 0.94× on pose) realistic for Tier 2, or are those numbers too optimistic?

### Risk & pragmatics

**Q9.** What's the **minimum-viable experiment** to validate the plan before committing 2-3 weeks of compute? E.g., is there a 1-2 day smoke test that would tell us whether the per-head upgrades are likely to work?

**Q10.** Given the AAIML deadline pressure (~3-4 weeks), **which tier should we actually do?** Tier 1 (cheap, 70% SOTA), Tier 2 (recommended, 80% SOTA), or Tier 3 (full, 95% SOTA)?

**Q11.** Is there a **Tier 0** option we're missing — something cheaper than Tier 1 that could still clear 70-80% SOTA? E.g., is there a "head upgrade only" path that doesn't require a backbone change?

### What we DON'T need (negative space)

**Q12.** Are there any **recommendations in files 188-190 that you disagree with**? Specifically:
- Should cross-task attention be tried anyway (despite your round-2 §2 Q7 saying skip)?
- Should we add MMoE despite the marginal upside?
- Should we add MixUp / heavy augmentation despite your round-2 §2 Q6 saying it likely hurts long-tail?

## Constraints (what you can assume)

- 2 GPUs available, 1 primary (1 in active use by current Path-D run)
- AAIML deadline ~3-4 weeks out
- Compute budget: ~15-20 GPU-days remaining
- License requirements: must clear AAIML publication (CC-BY-NC, MIT, or Apache 2.0 for any backbone weights)
- Code is in `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/`
- 4 Single-task training scripts don't exist yet — we need to write them
- Currently training on PID 2545563 (Opus 186 / Path-D run, will reach ep30 in ~10 hours)

## Deliverables expected

Please respond with:
1. **Direct answers** to Q1-Q12 above
2. **Validation or critique** of the 3-tier plan in file 189 §3
3. **A revised plan** if your answers change the recommendation (e.g., "skip YOLOv8, do this instead")
4. **A minimum-viable smoke test** we can run in 1-2 days to validate the per-head upgrades
5. **Any new strategic insight** we should consider

Please be direct. The honest read of the situation is more useful than encouragement.

---

## Closing notes (for the human who's pasting this)

- Files to attach or have Opus read:
  - `code/industreal_improved/analyses/consult_2026_06_10/AAIML/181_OPUS_MTL_PATH_DECISION_ANSWER.md` (Opus round 1)
  - `code/industreal_improved/analyses/consult_2026_06_10/AAIML/186_OPUS_ROUND2_STRATEGY_ANSWER.md` (Opus round 2)
  - `code/industreal_improved/analyses/consult_2026_06_10/AAIML/187_OPUS_181_186_IMPLEMENTATION_CHECKLIST.md` (status)
  - `code/industreal_improved/analyses/consult_2026_06_10/AAIML/188_PER_HEAD_ARCHITECTURE_UPGRADES.md`
  - `code/industreal_improved/analyses/consult_2026_06_10/AAIML/189_BACKBONE_AND_MTL_TOPOLOGY.md`
  - `code/industreal_improved/analyses/consult_2026_06_10/AAIML/190_TRAINING_PATH_AND_HYPOTHESIS_VALIDATION.md`
- Ground-truth files (read for verification):
  - `code/industreal_improved/scripts/train_mtl_mvit.py` (1563+ lines, post-Path-D)
  - `code/industreal_improved/src/models/mvit_mtl_model.py` (447 lines, post-PSR fix)
  - `code/industreal_improved/src/config.py` (NUM_DET_CLASSES, PSR weights, etc.)
  - `code/industreal_improved/analyses/consult_2026_06_10/AAIML/176_MTL_MViTv2_TRAINING_PROGRESS.md` (SOTA provenance)
  - **IndustReal paper**: arXiv:2310.17323 (Schoonbeek et al. WACV 2024) — please verify SOTA numbers from Tables 2 and 3 directly
- Verify against the *executed code*, not the prose in 176-180. Where docs and code disagree, the code wins (same authority rule as 181/186).

After you receive Opus's response, file 192 will capture the answer.

---

*This prompt was generated for Opus round 3. It assumes Opus has access to files 176-190 + the IndustReal paper PDF. The user is ready to paste this directly into a fresh Opus session.*
