# 173 — How to Prove "Multi-Task Helps" (efficient AND accurate, beats SOTA)

**Date:** 2026-07-08
**Premise (from the author):** "I am 100% fine changing the backbone, the architecture, or anything. I want to *prove* my hypothesis that multi-task is helping — even beat SOTA. We can make an MTL model that is more efficient *and* more accurate."
**This file:** the concrete architecture + experiment plan to do exactly that, grounded in what 172 established from the artifacts.

---

## 0. The one hard truth that decides everything

"Multi-task helps" is not a vibe; it is a **specific, falsifiable claim** that only exists relative to a control. There is exactly one way to prove it:

> Train each head **alone** and **together** on the **same backbone, same data, same schedule**, and show the together-version is **≥** the alone-version (accuracy), or **=** it at a fraction of the compute (efficiency).

Right now the repo has **zero** clean controls (172 §A1): the detection "single-task" is a *different architecture* (YOLOv8m vs ConvNeXt), and the activity "single-task" is a *frozen probe*, not a trained head. So today you literally cannot make the claim, no matter how good the MTL model is. **The single most important deliverable is not a better model — it is the matched single-task baselines.** Everything below is built so those baselines and the MTL model differ in *exactly one variable*: whether the other heads are present.

If you internalize nothing else: **the proof is the experimental design, not the architecture.** A modest model with clean controls proves the hypothesis; a brilliant model without controls proves nothing.

---

## 1. The claim you can actually win (pick the strong, provable version)

There are three tiers of "MTL helps." Claim the strongest you can support, and structure the paper so you fall back gracefully.

| Tier | Claim | Provability | What it needs |
|---|---|---|---|
| **T1 — Efficiency parity** | MTL matches single-task accuracy on all 4 heads at **N× fewer params / one forward pass** | **High** — almost certainly true with a shared backbone | matched ST baselines + measured FLOPs/params/latency |
| **T2 — Positive transfer** | MTL **beats** single-task on ≥1 head (auxiliary tasks add signal) | **Medium** — very likely for PSR (detection helps it) and plausibly activity | matched baselines + leave-one-out ablation |
| **T3 — Beats published SOTA** | MTL beats the published number on ≥1 head under **matched protocol** | **Medium** — winnable on activity and PSR, parity on detection | strong backbone + protocol-matched eval |

**Your headline sentence, if all three land:**
> "A single shared-backbone multi-task model matches or exceeds single-task accuracy on all four IndustReal heads — with **positive transfer** on procedure-step and activity recognition — while using **N× fewer parameters** and a single forward pass, and **exceeding published SOTA** on activity and step recognition under matched evaluation."

That is "efficient AND accurate AND beats SOTA," stated so every clause is backed by a measurement. **T1 is your floor and it is nearly free once the baselines exist. Do not gamble the paper on T3 alone.**

---

## 2. Why MTL *should* help on IndustReal (the mechanism — this is your scientific argument)

MTL helps when the tasks share real structure. On IndustReal they do, and you can name the structure precisely — this is the intellectual core of the paper, not hand-waving:

- **Detection → PSR is a hierarchy.** Procedure-step / assembly-state recognition *is* temporal aggregation of per-frame assembly-state detection. A model that must localize assembly states (detection) is building exactly the features PSR consumes. **Prediction: detection is a strong auxiliary for PSR — PSR-with-detection > PSR-alone.** This is your most likely positive-transfer result.
- **Activity ↔ PSR are cause and effect.** Actions ("plug wheel", "put brace") *cause* state transitions. Predicting the action and predicting the resulting step share mutual information; each regularizes the other.
- **Head pose is an attention prior.** Where the worker's head points indicates the manipulated region. Pose supervision biases the backbone toward the active workspace, which can help detection/activity focus — a soft spatial-attention auxiliary.

So the paper's "why" is: **IndustReal is a natural task hierarchy (localize → recognize action → infer step) plus an egocentric attention signal (pose), and MTL exploits that hierarchy.** That framing turns "we threw 4 heads on a backbone" into "we exploit the causal structure of assembly understanding." Reviewers reward mechanism.

---

## 3. Architecture: one shared hierarchical spatiotemporal backbone

The two-backbone V8 (YOLOv8m + MViTv2-S) **cannot** support the efficiency claim — two backbones is not sharing, it is two models in a trenchcoat (172 §B9). To be *efficient AND accurate* you need **one backbone whose features serve all four heads**. That requires a backbone that is simultaneously (a) **temporal** (for activity/PSR), and (b) **hierarchical/multiscale** (for detection).

**Primary recommendation: Hiera (MAE video-pretrained).**
- Hierarchical → produces multiscale feature maps (like a CNN), so you can hang a real detection FPN/neck on the early-stage high-res features. This is the property MViTv2-S/plain-ViT lack for detection.
- MAE video pretraining (K400/K600) → strong temporal features for activity and step transitions, and strong transfer with limited fine-tuning data (IndustReal is small — 3.5k clips).
- SOTA-competitive on **both** image detection and video classification, which is exactly the union you need.
- Good alternatives if Hiera integration is painful: **UniFormer-V2** (conv+attention hybrid, multiscale, strong video) or **MViTv2** *with an added multiscale detection neck*. Avoid plain ViT / VideoMAE-ViT for detection (single-scale).

**Head routing from the one backbone:**
```
Shared Hiera backbone  (clip [B,T,3,H,W])
├── stage-1/2 high-res spatiotemporal features ──► Detection neck (FPN) ──► per-frame boxes   (det head)
├── stage-4 pooled clip embedding              ──► Activity head (Linear/MLP)                  (act head)
├── sequence of per-clip embeddings            ──► Temporal PSR head (small causal transformer)──► transition logits (psr head)
└── stage-4 pooled embedding                   ──► Pose head (6D: fwd+up, renormalized)         (pose head)
```
Every head is **light** (the trainable surface today is only ~18M params — 172 §F). The backbone is ~90% of the model, so **sharing it is where the N× efficiency comes from**, and it is real and measurable.

**This is the "V9 unified backbone" you already gestured at in 169 — it is the right call, and it is what makes both halves of your claim (efficient + accurate) simultaneously true.**

---

## 4. Making MTL actually help (execution — this is where "wrong implementation" was killing you)

Interference *is* real (it is just second-order, per 172). Managing it is the "correct execution" that converts a collapsed model into one where MTL helps. In order:

1. **Fix the known defects first (non-negotiable, from 172):**
   - V8 `hash(cls_str) % num_classes` → stable ordered-dict class map (`train_v8_multitask.py:216`).
   - Activity zero-loss bug (`train.activity == 0.0`) → verify labels reach the loss and are not all-masked; assert `>0` gradient at step 0.
   - Empty-subsample detection eval (`det_n_present_classes==0`) → evaluate on full/GT-stratified val.
   - PSR head already repaired to LeakyReLU (`model.py:1604`); keep it.
   - No hard staging that zeroes heads — all heads on from epoch 1.
2. **Gradient balancing (the method contribution).** Uncertainty weighting (Kendall) alone is insufficient (it redistributes existing gradient, can't create signal). Add **gradient-conflict resolution**:
   - **PCGrad** (project conflicting task gradients apart) — simple, strong, cheap. Best first choice.
   - or **CAGrad / Nash-MTL** if you want a fancier headline method.
   - Combine with uncertainty weighting for scale. **This combination is a legitimate methods contribution** and directly operationalizes "MTL helps when optimized correctly."
3. **Per-head positive-signal sampling.** Sparse-positive heads (PSR transitions, rare detection classes) need batches that contain positives — a task-aware sampler so each head sees signal every step. This is what stops the small-gradient classification heads from being drowned.

---

## 5. The experiment that IS the proof (the controlled matrix)

Same backbone B (Hiera), same data, same schedule for every cell. This table *is* the paper's central result.

| Run | Det | Act | PSR | Pose | Purpose |
|---|---|---|---|---|---|
| ST-Det | ✓ | | | | single-task detection baseline |
| ST-Act | | ✓ | | | single-task activity baseline |
| ST-PSR | | | ✓ | | single-task PSR baseline |
| ST-Pose | | | | ✓ | single-task pose baseline |
| **MTL-All** | ✓ | ✓ | ✓ | ✓ | the multi-task model |
| LOO-noDet | | ✓ | ✓ | ✓ | does removing detection hurt PSR? (attributes transfer) |
| LOO-noPose | ✓ | ✓ | ✓ | | is pose a useful auxiliary or dead weight? |

**Read-outs:**
- **Transfer (accuracy):** per-head MTL-All vs ST-* → Δ per head, with bootstrap CI. Positive Δ on any head = T2 proven. (Expect PSR↑, activity ≈/↑, detection ≈, pose ≈.)
- **Attribution:** LOO rows show *which task helps which* (e.g., PSR drops in LOO-noDet ⇒ detection is what helps PSR ⇒ your hierarchy mechanism, §2, is confirmed empirically).
- **Efficiency:** params/FLOPs/latency of MTL-All vs sum of the four ST runs → T1. One backbone + 4 heads vs 4×(backbone+head).

This is a 5–7 run matrix. It is the difference between "we believe MTL helps" and "we measured that MTL helps, and here is the task-pair that drives it."

---

## 6. SOTA targets per head (realistic, and which are winnable)

| Head | Published anchor | Winnable? | Target | Lever |
|---|---|---|---|---|
| **Activity** | MViTv2-S 0.6223 (RGB) | **Yes — best "beat SOTA" bet** | 0.63–0.68 | stronger backbone (Hiera) + PSR/pose auxiliary transfer |
| **PSR (step)** | STORM / B (transition F1 — *resolve the 0.506-vs-0.901 conflict, 172 C-4*) | **Yes** | beat under matched transition-F1 protocol | detection-informed temporal head |
| **Detection** | WACV 0.838 (per-component, protocol-matched) | Parity, hard to beat | ~0.84–0.95 | Hiera + FPN; report protocol-matched, not native 0.995 |
| **Pose** | none | Automatic first baseline | 9.14° fwd / 7.78° up (with CI) | already have it |

**Beat-SOTA strategy:** don't try to beat SOTA on all four. Pick **activity and PSR** as the beat-SOTA heads (winnable with a better backbone + transfer), claim **parity** on detection, and **first-baseline** on pose. "Beats SOTA on 2 of 4, parity on 1, first baseline on 1, at N× efficiency" is a *stronger and more honest* paper than "beats SOTA on everything" (which no one believes anyway).

---

## 7. Order of operations (tiered by compute — tell me which tier you're in)

**Tier M — Minimum viable proof (fits a tight budget, ~1 GPU, days):**
1. Fix the 3 blocking bugs (§4.1).
2. Keep the *current* ConvNeXt (or MViTv2-S) backbone — don't swap yet.
3. Run the controlled matrix (§5) on that backbone. Even if absolute numbers are modest, **the MTL-vs-ST deltas prove the hypothesis.** T1 (efficiency) + T2 (transfer) are both reachable here. SOTA-beating (T3) probably not.
→ Outcome: a defensible "MTL helps (efficient + positive transfer) on IndustReal" paper.

**Tier F — Full ambition (multi-GPU / weeks):**
1. Everything in Tier M, plus:
2. Swap to the unified Hiera backbone (§3).
3. Add PCGrad/CAGrad gradient balancing (§4.2).
4. Re-run the matrix; push activity/PSR past SOTA (§6).
5. Protocol-match all SOTA comparisons (172 §C2/C8).
→ Outcome: "MTL is efficient AND accurate AND beats SOTA on activity+PSR."

Both tiers **start with the same first two steps** (fix bugs, run the matrix), so you can begin immediately and decide on the backbone swap once the matched-baseline harness exists.

---

## 8. Risks and the graceful fallback

- **Risk: no positive transfer on some heads (negative transfer is common in MTL).** Mitigation: T1 (efficiency parity) still holds and is publishable; report the honest per-head Δ (some +, some ≈) — that *is* the "cost of sharing" science, and it is what makes the paper credible rather than a sales pitch.
- **Risk: Hiera integration eats the whole budget.** Mitigation: Tier M proves the hypothesis on the existing backbone; the backbone swap is an accuracy upgrade, not a proof requirement.
- **Risk: SOTA protocol mismatch (splits/metrics differ).** Mitigation: 172 §C — pin splits and re-run at matched protocol before any "beats" sentence.
- **Non-negotiable:** the matched single-task baselines. Without them there is no proof, full stop.

---

## 9. What I need from you to start executing

The plan forks on one fact: **your compute budget and timeline.** That decides Tier M vs Tier F and whether we swap the backbone now or after the baselines. Once you tell me, I can start on the bug fixes and the controlled-matrix harness immediately — those are shared by both tiers and are the critical path to the proof.

*Bottom line: you don't need to abandon multi-task to prove it helps — you need matched controls, one shared backbone to make efficiency real, gradient balancing to make transfer real, and honest per-head deltas. The hypothesis is winnable; the missing piece was never the idea, it was the controlled experiment.*
