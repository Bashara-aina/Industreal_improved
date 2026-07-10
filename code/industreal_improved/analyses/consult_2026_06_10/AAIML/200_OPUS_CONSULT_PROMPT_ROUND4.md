# 200 — Opus Consultation Prompt: Round 4

**Date:** 2026-07-10
**Previous rounds:** 181 (Path-D), 186 (corrections), 192 (Tier A)
**Status:** run11 LIVE (PID 885592), architecture upgraded after EP10 evidence

---

## Summary for Opus

Since Round 3 (file 192), we:

1. **Implemented all Tier A recommendations** (16 scripts, subject-overlap verified, checkpoint verification, TAL assigner, PSR T=8 native, detection P2 skip, model soup, 4 ST baselines ready, E8 diagnostic ready)

2. **Killed run10** after epoch 10 eval showed: detection 0.0 mAP, activity 0.58% top-1, PSR F1 0.004. Three of four heads at exactly zero.

3. **Diagnosed the cause:** Heads were architecturally inadequate, not that MTL fundamentally fails:
   - Activity: 2-layer MLP (1.1M) can't discriminate 75 classes from single cls_token
   - Detection: Sparse 3×3 from semantics-free P2 conv_proj features
   - PSR: 4-layer transformer on 96-dim conv_proj features (no semantic signal)

4. **Upgraded all three heads** and relaunched as run11 (117.7M total, was 46M):
   - Activity: 3-layer MLP (768→2048→1024→75), GELU, dropout 0.2 (3.75M)
   - Detection: TAL (TOOD) topk=10, P5/P4/P3 only, skip P2 (4.5M)
   - PSR: 6-layer causal T, d=768→6144 ff, P5 features, native T=8, Focal-BCE (70.9M)
   - Pose: unchanged (already healthy at 9°)

5. **Got immediate positive signal:** PSR loss dropped from 1.56 to 0.17-0.25 (10×). Detection alternates 0.001/4-5 (TAL providing real gradient). Activity loss still high (4-5) but training.

---

## The Question

**Given our current state (run11 training, first eval at ep10 in ~6 hours), what is the optimal path to produce a paper that proves "MTL helps, not hurts — it is more efficient and at least as accurate across all heads"?**

We need your strategic judgment on:

1. **Should we commit to Path B (ST baselines) or Path C (soup) regardless of EP10 numbers?**
2. **If EP10 shows low numbers on some heads, what's the triage order? (Pose first, then PSR, then detection, then activity?)**
3. **Is the 117.7M model viable for a paper, or is 70.9M for PSR alone indefensible?**
4. **Can we claim "more efficient" with a 117.7M model vs ~120M of specialists? The efficiency ratio is ~1.0×, not 3×. Does the latency argument (one forward pass vs four) save this?**
5. **What's the minimum viable paper if MTL doesn't beat ST on any head? Is the Kendall-collapse + bounded-cost story enough?**

---

## The Detailed Evidence

### File 195: Overview (this round's context)
→ Read for: document map, what changed since Round 3

### File 196: Architecture Deep Dive
→ Read for: full model spec, feature flow, param counts, training protocol, Kendall caps, PCGrad

Key facts:
- 117.7M total (34.5M backbone + ~83M heads)
- PSR head is 70.9M alone (60% of total)
- P5 features for PSR (768-dim semantic, was 96-dim conv_proj)
- TAL assigner from TOOD (ICCV 2021), topk=10, cited properly
- P2 skipped (detection uses P3/P4/P5 only)
- Focal-BCE for PSR (default on), sqrt-tame inv-freq for activity
- Kendall caps: act≤1.0, psr≤0.5, det≤1.5, pose≤2.0

### File 197: Results & Metrics
→ Read for: run10 EP10 data, run11 batch losses, expected EP10 outcomes, eval protocol

Key facts:
- run10 EP10: det 0.0, act 0.58%, psr 0.004, pose 8.9°
- run11 detection: alternating 0.001 (no-GT) / 3.9-4.8 (GT) — real signal
- run11 PSR: 0.15-0.25 loss (was 1.56) — breakthrough
- run11 activity: 4.3-5.6 loss — still high
- run11 pose: 0.01-0.05, occasional single-batch spikes
- First eval ~14:00-15:00 JST

### File 198: Per-Head Analysis
→ Read for: SOTA comparison, bottlenecks, realistic ceilings, per-task deep dives

Key facts:
- Detection SOTA: 0.779 (IndustReal-only), 0.838 (with synthetic)
- Activity SOTA: 0.6525 (MViTv2-S + single linear, single-task)
- PSR SOTA: 0.883 (STORM, fundamentally different paradigm — transition detection from ASD outputs, not per-frame component prediction)
- Pose: No SOTA — our contribution (first baseline)
- Activity: cls_token may be overloaded for 4 tasks (single 768-dim vector serving activity, pose, and indirectly detection/PSR)
- Activity SOTA was reached with single-task MViTv2-S — same backbone, but dedicated to one objective

### File 199: Path Options & Decision Matrix
→ Read for: 5 strategic paths with costs, risks, expected outcomes

| Path | Time | Risk | Quality | ST Required |
|------|------|------|---------|-------------|
| A: run11→ep30→paper | 2 days | Medium | Good | ❌ |
| B: A + ST baselines | 5-7 days | Low | Strong | ✅ |
| C: B + Soup + finetune | 7-10 days | Low-Med | Strongest | ✅ |
| D: Scale backbone | 10-14 days | High | Unclear | N/A |
| E: Per-head triage | 3-5 days | Medium | Variable | ❌ |

---

## Specific Questions for Opus

### Q1 — The Efficiency Claim with 117.7M

File 192's FC-3 correctly identified that 4 real specialists (MViTv2-S + YOLOv8-m + STORM + pose MLP) ≈ 120M total. Our current model is 117.7M — essentially the same parameter count. The efficiency argument shifts from "fewer parameters" to "single forward pass = lower latency / less memory."

**Is "one backbone forward pass for 4 tasks" a defensible efficiency claim when total params are similar? Or do we need to reduce head parameters to make the claim credible?**

We could reduce PSR from 6 layers to 4 (saves ~24M) or from d=768 to d=512 (saves ~40M). But the PSR signal is finally working — reducing capacity risks killing it.

### Q2 — The PSR Parameter Explosion

PSR head is 70.9M of our 117.7M model. The previous 3M head was non-functional. Is there a middle ground that's both functional AND defensible?

Options:
- Keep 6 layers, reduce d_model from 768 to 512 (70.9M → ~30M)
- Keep d_model=768, reduce to 4 layers (70.9M → ~47M)
- Keep 3 layers but increase d_model to 768 (was 3 layers at d=96) — smallest change from old

### Q3 — Activity: Keep 3-Layer MLP or Add VideoMAE Stream?

Activity at 0.58% EP10 (run10) is below random. The 3-layer MLP in run11 adds capacity but the fundamental bottleneck — one cls_token for 75 classes across 16 frames — remains.

Option: Enable VideoMAE V2 stream (frozen, +22M params, +600MB VRAM). The config already has USE_VIDEOMAE support but it's disabled. Opus 192 Q3 said "trust the 2-layer MLP" — but that was when we thought the 2-layer MLP would reach 5-10% by ep10.

**At what activity threshold should we consider adding VideoMAE? If EP10 shows <3% top-1, is that the trigger?**

### Q4 — Detect Augmentation

`--det-aug` (flip+color+crop) is implemented but NOT active in run11. Opus 192 Q6 said "do mosaic on detection." Should we activate detection augmentation now (requires restart) or wait for EP10 evidence?

### Q5 — ST Baselines: Parallel or After run11?

GPU 2 (16GB) is free. We could launch ST baselines now while run11 trains. But:
- ST detection training might interfere with run11 if both use GPU resources
- ST baselines use the same training script but --task flag — need to verify they work with current code
- PSR ST baseline: what head architecture? The 70.9M P5 head is MTL-specific. Should ST PSR use a smaller head?

**Should we launch ST baselines now, wait for EP10, or wait for run11 to complete?**

### Q6 — The "Honest Miss" Story

Opus 192 Q7 and 186 G-2 both say PSR is the pre-registered honest miss: "our per-frame approach reaches X% of the specialist pipeline while sharing a backbone." But if PSR turns out to be the one head that DOES work (because of the 10× loss drop), does that change the paper narrative? The "honest miss" might become "surprising strength," and detection or activity might be the actual miss.

### Q7 — Minimum Viable Paper

If EP10 still shows 0.0 on 2+ heads, and ST baselines are much better, what's the minimum viable paper?

**Option 1 — Kendall Pathology Paper:** "We characterize and fix a degeneration in Kendall uncertainty weighting for multi-task learning. The fix prevents the highest-loss task from being starved. We demonstrate the collapse on a challenging 4-task assembly understanding benchmark, showing it reduces loss variance from X to Y and enables all heads to learn where uncapped weighting fails entirely."

- Requires: run11 to show improvement over prior runs (which it already does — PSR 1.56→0.17)
- Does NOT require: beating SOTA, positive transfer, MTL/ST > 1.0
- Risk: "You just tuned hyperparameters" — reviewer needs convincing this is methodology, not tuning

**Option 2 — Efficiency Trade-off Paper:** "Sharing a backbone across 4 tasks trades accuracy for inference speed. At 3× lower latency, our model retains X% of per-task specialist accuracy. We characterize per-head transfer effects via gradient heatmaps."

- Requires: non-zero numbers on all heads, ST baselines for comparison
- Does NOT require: MTL > ST on any head
- Risk: "Everyone knows MTL trades accuracy for efficiency" — not novel

**Option 3 — Positive Transfer Claim (current bet):** "On pose estimation, multi-task learning provides positive transfer, improving over the single-task baseline by X°. On three other tasks, the shared backbone achieves Y% of specialist performance. The total system uses Z× fewer parameters than running four specialists."

- Requires: MTL > ST on at least pose
- Risk: If ST pose is also ~9°, the positive transfer claim collapses to "no cost" on pose

**Which of these three options is the strongest paper if the ideal (MTL > ST on pose, MTL ≈ ST on others) doesn't materialize?**

### Q8 — The MViTv2-S Ceiling

Your FC-6 (file 192) correctly identified that the IndustReal-only detection ceiling is 0.779 (not 0.838). Similarly, the activity SOTA of 0.6525 was set by single-task MViTv2-S.

**If single-task MViTv2-S achieves 0.65 activity, what is the theoretical ceiling for MTL MViTv2-S?** Can a shared backbone mathematically reach the same representations as a dedicated one, or is there an irreducible information loss from multi-task optimization?

This matters for framing: if the ceiling is ~0.5 (say), then MTL reaching 0.4 is 80% of ceiling — much stronger than 60% of 0.65.

---

## What We Need From Opus

A strategic decision on the path forward, ideally with:
1. **Primary recommendation:** Which path (A/B/C/D/E/hybrid) and why
2. **EP10 contingent plan:** "If EP10 shows X, do Y. If EP10 shows W, do Z."
3. **Parameter budget verdict:** Is 117.7M absurd or defensible? Should we reduce PSR?
4. **Paper framing:** Which of the three minimum-viable-paper options (Q7) is strongest given our evidence
5. **ST baseline timing:** Launch now, wait for EP10, or wait for run11 completion
6. **Activity strategy:** Threshold for VideoMAE vs trust 3-layer MLP
