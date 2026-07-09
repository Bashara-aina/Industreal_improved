# 192 — Opus Round 3 Answer: Validating (and Correcting) the Plan to Make MTL Beat SOTA

**Date:** 2026-07-09
**Inputs read line-by-line:** 176 (progress + SOTA provenance), 181 (Opus round-1), 186 (Opus round-2), 187 (implementation checklist), 188 (per-head upgrades), 189 (backbone + MTL topology), 190 (training path + hypothesis validation), 191 (round-3 prompt).
**Ground-truth re-verified against the executed code, this round:**
`src/models/mvit_mtl_model.py` (490 lines, post-PSR-P5 fix), `scripts/train_mtl_mvit.py` (detection loss, FPN levels, grad-accum, log-var caps).
**Authority rule (unchanged from 181/186):** where the plan documents (188–190) and the code/SOTA-provenance disagree, the code and the provenance win, and I say so explicitly. Files 188–190 were written *after* 186; several of their premises quietly reverse 186's conclusions without new evidence. I flag every place that happens.

---

## 0. READ THIS FIRST — the one-paragraph answer

**Files 188–190 are a well-engineered answer to the wrong question.** They ask *"what architecture will let one shared backbone beat 80–95% of four specialists' SOTA?"* — and then reach for exactly the machinery (a 300M frozen foundation backbone, a from-scratch YOLOv8 head, a "STORM-like" decoder, ArcFace, model soup, three escalating tiers) that **round 2 already refuted on primary-source evidence.** The refutation still holds and nothing in 188–190 overturns it: the activity SOTA (0.652) *is* MViTv2-S with a single linear head; the detection SOTA (0.838) is a *smaller* CNN (YOLOv8-m) and *requires synthetic-data pretraining* — IndustReal-only is 0.779 boxed / 0.575 entire-video; PSR's 0.901 is a purpose-built state-tracker. The gap to a good paper is **not representational capacity, and it is not four missing head redesigns.** It is three things, in this order: **(1) you have not yet proven the current 0.0 / 0.008 / 0.0 numbers are real** — they are contaminated by fresh-initialised heads (the 2-layer activity MLP and the 768-ch PSR head were *re-initialised on resume* per checklist B-9, so ep7 is effectively **ep0** for them) and by a **detection eval that a *prior* run showed reaching 0.468 mAP** (176 §3.4); **(2) two of the four heads have a genuine, cheap, code-level defect that no amount of new architecture fixes** (PSR pools T=16→8 then *linearly interpolates* 8→16, so it literally cannot represent a sharp transition; detection's finest level P2 reads raw `conv_proj` patch-embeddings — the same semantics-free features that starved PSR); **(3) the paper's actual contribution — the Kendall-collapse characterisation + fix + the MTL-vs-matched-ST efficiency comparison — does not require beating SOTA at all, and the foundation-backbone plan actively destroys it** (it makes the model 300M+, which *inverts* the efficiency claim: your four *real* specialists total ~120M, not the 400M file 190 §6.1 assumes). **My recommendation: do NOT do Tier 1/2/3 as written. Run a 1–2 day minimum-viable smoke test (§6) to find out which "0.0"s are artifacts, keep MViTv2-S, fix the 2–3 real bugs, run the 4 single-task baselines you need anyway, try the near-free soup init, and publish L2+L3+method honestly.** That is a genuinely strong, reviewer-proof paper — the Kendall-pathology story is novel — and it fits the deadline with margin. Chasing "beat SOTA on all four in one 40M model" is the trap 186 named; it will consume the three weeks and produce a *weaker* paper.

---

## 1. THE CENTRAL STRATEGIC ERROR IN 188–190 (five layers)

This is the single most important section. Everything in Q1–Q12 flows from it.

**Layer 1 — The surface framing.** Path-D reaches ~50–60% of SOTA; 80% "requires representational + capacity" upgrades (191, "honest assessment"). So: upgrade the heads and the backbone. This is intuitive and wrong.

**Layer 2 — The premise was already falsified.** 186 §0 downloaded arXiv:2310.17323 and read Tables 2–3 directly. Activity SOTA = MViTv2-S / 65.25% top-1 — *your exact backbone class, with a single linear classifier.* Detection SOTA 0.838 = YOLOv8-m (~25M, **smaller** than your 34.5M backbone) **with IndustReal+synthetic**; IndustReal-only is 0.779 boxed / 0.575 entire-video. PSR 0.901 = STORM, a dedicated tracker. **None of the three SOTAs was set by scaling capacity.** So "the remaining gap is representational + capacity" (191) is contradicted by the very table you are chasing. 188–190 never cite new evidence against this; they simply assume capacity is the bottleneck again. It is not.

**Layer 3 — Two goals are being conflated.** There are two different papers hiding in these documents:

- **Goal A (what 181 and 186 said is publishable):** one small shared backbone does 4 tasks; measured against *matched single-task baselines you train*, it shows positive transfer on ≥1 head, ~90% on the benign ones, big parameter/inference efficiency, plus the Kendall-collapse method fix. **This does not require beating any specialist SOTA.**
- **Goal B (what 188–190 chase):** the shared model beats 80–95% of four *different* specialists' absolute SOTA simultaneously.

Goal B is strictly harder than any one of those SOTAs (each was set by a different purpose-built system) and — this is the load-bearing point — **the standard tool for Goal B (a 300M foundation backbone) is fatal to Goal A.** You cannot have both. 188–190 try to, and in doing so they quietly abandon Goal A's efficiency thesis.

**Layer 4 — The efficiency claim inverts, and file 190 miscounts it.** File 190 §6.1 claims "MTL (frozen foundation + LoRA) = 309M is ~26% more efficient than 4 ST specialists (400M)." The 400M is invented — it assumes each specialist is 100M. **The real specialists on *this* dataset are MViTv2-S (34.5M) + YOLOv8-m (~25M) + STORM (small, ≈ tens of M) + a pose MLP.** Four real specialists ≈ **~120M total**, not 400M. Against ~120M, a 309M "efficient MTL" model is **2.6× larger.** The efficiency headline doesn't weaken — **it reverses sign.** The *only* configuration in which "MTL is more efficient/faster" is true is the one you already have: MViTv2-S, 43.5M total, one forward pass ≈ **2.7× fewer params than the real specialists** and genuinely lower latency (190 §6.3's "2–3× lower inference latency" is true for MViTv2-S and false for a 304M backbone that is ~10× slower than MViTv2-S — 189 §1.3 admits this in its own "Cons"). So the backbone swap does not trade "some efficiency for accuracy"; it deletes the paper's thesis.

**Layer 5 — What is *actually* limiting the current numbers.** Strip away the capacity narrative and look at the code + provenance:

| "Symptom" (191 table) | What it actually is | Fix class |
|---|---|---|
| Detection 0.0 mAP | (a) heads/loss recently changed; (b) a *prior* run reached **0.468** (176 §3.4) → **eval-harness / target-encoding is a live suspect**; (c) finest level P2 reads raw `conv_proj` semantics-free features | Verify eval **first**; then feature-source + TAL-lite. **Not** a from-scratch head. |
| Activity 0.008 | The 2-layer MLP is **fresh-init on resume (B-9)** → ep7 ≈ ep0; loss "flat at 4.83" is a *cold* head, not a stuck one | Let it train; ST-activity(clip) ceiling check. **Not** ArcFace/attention-pool. |
| PSR 0.0 F1 | (a) P5 head also **fresh-init on resume (B-9)**; (b) **T=16→8→16 interpolation caps temporal resolution**; (c) rare-event BCE; repo shows PSR reaching **0.347** elsewhere (186 §1) | Temporal-resolution + loss fix. **Not** an unverified "STORM-like" decoder. |
| Pose 10° | Healthy, likely positive-transfer win | None. |

**Conclusion of §1:** The correct move is *diagnosis before architecture*. The plan in 188–190 spends 8–12 engineering days + a backbone swap to solve problems that are, in at least two cases, fresh-init + eval artifacts, and in the other two, **one-file bug fixes**. Do the diagnosis (§6) before committing.

---

## 2. FACTUAL CORRECTIONS (code and provenance win)

In the tradition of 181 §1 and 186 §5 — new this round, all code-grounded.

**FC-1 — There is no "YOLOv8 paper."** File 188 §1.3/§9.1 says "re-implement from the YOLOv8 paper (`YOLOv8_Scaled_YOLOv8_CVPR2023`)"; 189 §8 cites "the YOLOv8 paper." **No such paper exists** — YOLOv8 (Ultralytics, 2023) was never published as a peer-reviewed paper, and `YOLOv8_Scaled_YOLOv8_CVPR2023` is not a real citation. The *components* you want have real, citeable sources: **TAL** = Feng et al., *TOOD: Task-aligned One-stage Object Detection*, ICCV 2021; **DFL** = Li et al., *Generalized Focal Loss*, NeurIPS 2020; **decoupled head** = Ge et al., *YOLOX*, 2021. Re-implement and cite *those*. This also dissolves the AGPL worry: TAL/DFL are described in Apache/MIT-adjacent academic papers; you never touch Ultralytics code.

**FC-2 — The decoupled head already exists.** File 188 §1.2 point 2 states "No decoupled head … the current code uses a single `reg_preds` head shared with classification features." **False.** `mvit_mtl_model.py:186–197` defines `DetectionHead` with a **separate `cls_head`** (Conv→BN→ReLU→Conv→24) and a **separate `reg_head`** (Conv→BN→ReLU→Conv→4·reg_max). The classification and regression branches are already decoupled. So one of the three "structural issues" motivating the YOLOv8 rewrite is not present in the code. Only the *assigner* (TAL) is genuinely missing.

**FC-3 — The efficiency math is wrong and inverts under the foundation backbone.** (See §1 Layer 4.) File 190 §6.1's "4 × 100M = 400M" specialist baseline is fictional; the real per-task specialists total ~120M. Any headline that pairs "efficient" with a 304M frozen backbone is indefensible and a reviewer will catch it in one line.

**FC-4 — PSR is not intrinsically 0.0, and it has a temporal-resolution bug.** 186 §1 already noted the repo's `eval_results.json` shows PSR F1 0.347 / 0.441 on a different pipeline. New this round: the P5 PSR head (`mvit_mtl_model.py:313–337`) **spatial-pools P5 (7×7→1×1), then interpolates the temporal axis T=8→16 by `mode="linear"`.** The backbone temporally pools T=16→8; PSR upsamples 8→16 with linear blends. **A task whose entire label is a sharp per-frame transition is being asked to predict it through a linear temporal interpolation** — odd frames are blends of neighbours, so a transition between frames 2k and 2k+1 is unrepresentable at full sharpness. This is more load-bearing than transformer-vs-GRU and is invisible in 188 §3.

**FC-5 — Per-frame tokens are NOT exposed.** Verified: `MViTFeaturePyramid.forward` returns `(fpn_features, cls_token)` only; the sequence tokens `x[:, 1:, :]` are consumed *inside the FPN hooks* and never surfaced as `[B, T, 768]`. So 188 §2.4 / §5's temporal-attention-pool activity head **cannot be built without re-plumbing the backbone forward** — exactly what 186 Q3 said. 188 §5 calls this "0.5 day"; it is delicate (the reshape must respect MViT's pooled T=8, not T=16), and it buys little (see Q3).

**FC-6 — Recompute the "80% bars" against the *IndustReal-only* ceiling.** The 0.838 detection SOTA needs synthetic pretraining you do not have. The honest reference for a model trained on IndustReal alone is **0.779 boxed / 0.575 entire-video** (186 §0, verified). So a shared-backbone detector should be judged against ~0.78 (boxed) — 80% of that is **0.62**, not 0.67 — and against 0.575 (entire-video) — 80% is **0.46**. Chasing 0.67 against a synthetic-augmented number your training set can't reach is self-sabotage.

**FC-7 — "STORM-like" is unverified branding.** 186 §1 flagged STORM "is not even in the original IndustReal paper." File 188 §3.3's `STORMDecoder` is a generic 2-layer GRU + linear head; nothing about it is verified to match STORM's actual method, and §3.2's own "detect transitions as spans" idea is **not implemented** by §3.3's per-frame decoder (internal inconsistency). Before targeting 0.901 or claiming to emulate STORM, confirm what STORM is and, critically, **what its F1@k metric definition is** — a metric mismatch invalidates the comparison (186 H-1/2/3).

---

## 3. DIRECT ANSWERS — Q1–Q12

### Q1 — Is the YOLOv8-head re-implementation the right call for detection? Is TAL the load-bearing change for 0.0 → 0.67?

**No, TAL is not the load-bearing change, and a from-scratch YOLOv8 head is over-investment.** Five layers:

1. *Surface:* TAL gives 10–50 dense positives/GT vs your 9 (3×3); denser supervision → better mAP. True but incremental.
2. *Density is already there.* Detection runs at **four** levels including **P2 at stride 4 = 56×56 = 3136 cells** (plus 784+196+49). You are not positive-starved the way a 3-level YOLO neck is; 181's own estimate for the 3×3 fix was 0.1→0.3, and TAL on top is a further increment, **not** a 0.0→0.67 jump.
3. *0.0 is suspicious.* A prior ConvNeXt-MTL run reached **0.468 mAP** (176 §3.4), and 176 explicitly attributes an earlier "det 0.0" to an **eval subsample artifact.** Before rewriting anything, confirm the eval harness returns non-zero mAP on a model that provably detects (overfit 200 images — §6). If the 0.0 is an eval/target-encoding bug, no head redesign fixes it and the rewrite wastes 4 days.
4. *The real feature problem.* P2 = raw `conv_proj` patch-embeddings (semantics-free — the exact issue that starved PSR). TAL reassigns *which* cells are positive; it does not improve *what* those cells read. The higher-leverage change is the **feature source** (run detection classification on the semantic levels P3/P4/P5, or on per-frame features rather than a temporal mean) — TAL cannot compensate for layer-0 features.
5. *Decision.* The decoupled head already exists (FC-2). So the genuine, bounded upgrade is: **port only the TAL assigner + the DFL/CIoU/BCE loss package (~250 lines, cited to TOOD/GFL/YOLOX, no AGPL) onto the existing FPN+decoupled head, and move detection off the raw-P2 feature source.** ~2 days, lower risk than a from-scratch rewrite that re-introduces the two bugs 186 already found. **Judge results against the IndustReal-only ceiling ~0.78 boxed / 0.575 entire-video (FC-6), and frame the paper claim as "shared-backbone MTL reaches X% of the YOLOv8-m specialist,"** exactly as 186 Q5 said. **Bottom line: TAL is worth doing but it is a second-order lever; eval-verification and feature-source are first-order.**

### Q2 — Is the STORM-like decoder right for PSR, or is the bottleneck elsewhere?

**The bottleneck is elsewhere; do not build an unverified "STORM-like" decoder.** GRU-vs-transformer is irrelevant — both model T=16 fine. The real bottlenecks (FC-4): (a) the **T=16→8→16 interpolation** caps temporal resolution below the transition sharpness the metric rewards; (b) **rare-event BCE** plateaus at base-rate; (c) the P5 head is **fresh-init at ep7** (B-9), so "flat at 1.58" is a cold head, not a stuck one; (d) the F1@k **metric/target definition** is unverified against STORM. The right sequence: **first get a non-zero baseline** (the repo pipeline reached 0.347 — reproduce that eval on the P5 features), **then** fix temporal resolution (predict at the backbone's native T=8 and evaluate there, or upsample features *before* the temporal encoder rather than after — do not linearly blend the outputs) and **swap BCE for a rare-event loss** (focal-BCE or an asymmetric/ F1-surrogate loss). Only if PSR is *still* the miss after those cheap fixes do you consider a dedicated decoder — and even then, **PSR is the pre-registered honest miss** (186 G-2), so accept it rather than spending 3–5 days chasing 0.72. A GRU whose §3.2 rationale (spans) it doesn't even implement (FC-7) is not that fix.

### Q3 — Temporal attention pool + ArcFace for activity, or trust the 2-layer MLP?

**Skip both; trust the 2-layer MLP and let it train.** The decisive evidence is the benchmark itself: **the 0.652 activity SOTA was reached with MViTv2-S + a single linear layer + plain CE** (186 §0). Plain CE on the pooled class token is *demonstrably sufficient to hit SOTA*, so the head is not the bottleneck. Three problems with the proposed upgrades: (i) the current 0.008 is a **fresh-init head at ep0** (FC), not evidence the MLP is inadequate; (ii) **temporal-attention-pool requires re-plumbing** (FC-5) and re-pools features MViT's 16 blocks *already* attention-pooled into the class token — near-redundant, low expected value; (iii) **ArcFace is unproven for closed-set 75-class long-tail** and margin-sensitive (188 admits m=0.30 is fragile). If, after ST-activity(clip) establishes the ceiling, MTL activity lags, the *principled, cheaper* long-tail tool is **balanced-softmax / logit-adjustment** (Menon et al. 2020) — a drop-in bias correction — not ArcFace, and not attention-pool. Note the residual MTL gap is a **shared-backbone compromise**; neither ArcFace nor attention-pool addresses that (only better MTL weighting / soup init does). So they solve the wrong problem. **Consistent with 186 Q3.**

### Q4 — Frozen foundation backbone (InternVideo2-L / DINOv2-L) or scale MViTv2 S→L?

**Neither as the headline. Keep MViTv2-S.** The foundation backbone is the single worst idea in 188–190 for *this* paper: it is unnecessary (capacity isn't the bottleneck — §1 L2), it **inverts the efficiency claim** (FC-3), it **breaks apples-to-apples comparison** with the WACV MViTv2-S baseline that makes your MTL number legible (186 Q1), it **destroys the "faster inference" claim** (304M ≈ 10× slower than MViTv2-S — 189 §1.3's own Cons), and it carries **license risk** (InternVideo2 weights ≠ clean Apache — 186 §5.3). MViTv2-L (53M, CC-BY-NC — acceptable for AAIML per your constraints) is a *mild, license-clean hedge* (186 B-3) but addresses a non-binding bottleneck at 2× compute; spend that compute on **data coverage + soup init** first. **The only legitimate place for a 300M backbone is a single-head *headroom ablation*** ("with a 300M backbone, activity reaches X; the shared 40M model captures Y% of that at 1/7 the params"), reported as an ablation, never the headline — exactly 186's demotion of Strat-2.

### Q5 — Sequential pretraining + model soup (Strat-4) worth 2–3 weeks?

**Yes to the soup; no to the "2–3 week Strat-4" framing — the framing double-counts work you must do anyway.** The 4 single-task pretrainings are **not** Strat-4 overhead; they are the **baselines the paper cannot exist without** (181 §4, 186 §2). Given you're running them regardless, the *marginal* cost of the soup path is ~5 minutes (weight averaging) + one finetune. So do it — but understand what it buys: **a warmer MTL init** where every head starts near-competent, which *reduces the Kendall-starvation dynamics early* (no single cold loss dominates). That's a real, cheap benefit. Caveat (deep): Wortsman soups average models fine-tuned on the *same* task; averaging backbones fine-tuned on *four very different* objectives may land between basins and be worse than any (190 §9 admits this). So **soup is a nice-to-have init, not a dependency** — a 5-minute experiment tells you if the soup'd backbone's loss is lower than cold init; if not, drop it. **Net: run the mandatory baselines, try the soup as a cheap increment, don't call it a 3-week tier.**

### Q6 — Is MTL/ST ratio ≥0.9 across all 4 heads the right "MTL is helping" threshold?

**No — it is falsely precise and conceptually off.** "MTL is *helping*" means **positive transfer**: ratio **> 1** on ≥1 head. Ratio ∈ [0.9, 1) is *benign sharing* (small MTL cost bought back by efficiency), not "helping." A single global ≥0.9 pass/fail also ignores that (i) metric CIs at 3 seeds are often ±0.02–0.05, so 0.90 vs 0.95 vs 1.00 are frequently statistically indistinguishable, and (ii) the right bar is **task-dependent** (pose: MTL ≤ ST MAE = positive transfer; PSR: even 0.8× is a fine efficiency story). **Recommendation:** drop the global threshold as a gate. Report per-head MTL/ST **ratios with CIs**; headline the heads where ratio > 1 (positive transfer), report efficiency independently of any ratio, and state the honest cost where ratio < 1. This is precisely 181/186's **L2+L3+method**. A ≥0.7 "still worth it for efficiency" line is fine for *your internal* go/no-go, not for the paper as a bar.

### Q7 — If MTL is 0.85× ST on one head (e.g. PSR), is there still a defensible paper?

**Yes — and PSR at ~0.85× is the *expected, pre-registered* outcome, not a failure** (186 G-2 predicted it). A single soft head breaks the hypothesis *only if the hypothesis is mis-framed as "beat ST on all four."* Framed as L2+L3+method it is robust to 1–2 soft heads provided: (i) ≥1 head shows positive transfer (pose is the likely clean win; detection is a candidate), (ii) efficiency holds (⇒ **keep MViTv2-S** — the thing that would actually break the paper is the backbone swap, not a soft PSR — FC-3), (iii) the Kendall-collapse method contribution stands. **The strongest move turns the soft head into a finding:** pair it with the E8 gradient-cosine heatmap (181 §4, 186 J-2) to show *which* task pairs conflict and why. A characterised negative transfer is science, not a hole. **So: fully defensible; pre-register PSR as the miss now and build the narrative around per-task transfer.**

### Q8 — Are the headline-table ratios (0.96/0.97/0.97/0.94) realistic for Tier 2?

**Treat the table as a hypothesis to measure, not a forecast — and it is optimistically smooth.** Two problems. (a) *The absolutes are shaky:* ST-activity 0.60 is grounded (SOTA 0.652 is MViTv2-S single-task); but **ST-detection 0.75 is optimistic** for a shared *video* backbone (a specialist YOLOv8-m gets 0.779 on IndustReal-only and your temporal-pooled MViT features will likely undershoot it — 186 Q5), and **ST-PSR 0.70 is optimistic** (repo pipeline reached 0.347–0.441; 0.72 is your single hardest bar). Expect ST-det ~0.55–0.70, ST-PSR ~0.45–0.65. (b) *The ratios are too uniform:* real MTL is lumpy — pose likely **> 1.0** (positive transfer), PSR likely **0.8–0.9**, activity/detection in between. An all-heads-≈0.95 table reads as a wish, not a prediction. **Do not put predicted numbers in the paper; put measured ones.** The *ratios* are more backbone-robust than the absolutes, which is another reason to keep MViTv2-S and stop tuning the absolutes toward a synthetic-augmented SOTA you can't reach (FC-6).

### Q9 — Minimum-viable experiment (1–2 days) before committing 2–3 weeks?

**Yes — and it is the most important thing in this document. It is a diagnosis suite, not a training run.** Its job is to find out *which "0.0"s are artifacts* so you don't spend three weeks fixing non-problems. Full protocol in §6; the four probes, in priority:

1. **Overfit-200-images per head → does the *eval metric* move?** (½ day, the #1 probe.) If a head can drive train loss to ~0 but the eval mAP/F1/top-1 stays at 0, the bug is the **eval harness / target encoding**, not the architecture — and the 0.468 prior detection result (176 §3.4) makes this a live hypothesis. Nothing in 188–190 helps if this is the cause.
2. **ST-activity(clip), 5 epochs.** (½ day.) ≥0.30 by ep5 ⇒ head+backbone fine, gap is MTL ⇒ *don't* touch the activity head.
3. **PSR on P5, reproduce the 0.347 eval + temporal-resolution A/B.** (½ day.) Non-zero F1 quickly ⇒ STORM decoder unnecessary.
4. **Detection TAL-lite vs 3×3 on the overfit set.** (½ day.) If 3×3 already overfits 200 images to mAP ≥ 0.6, the assigner isn't the bottleneck — features/eval are.

**Decision rule:** commit to an expensive upgrade *only for a head whose MVP probe shows the cheap fixes are exhausted.* Most likely, ≥2 of the four "0.0/0.008"s dissolve here.

### Q10 — Which tier for AAIML (~3–4 weeks)?

**None of Tier 1/2/3 as written. Do the re-scoped plan (§5) — call it "Tier A."** It keeps MViTv2-S, runs the MVP, lets Path-D finish, runs the 4 mandatory baselines, tries the soup init, fixes the 2–3 real bugs, and publishes L2+L3+method. It is **cheaper than their Tier 1** (no MViTv2-L scale-up, no from-scratch heads — targeted fixes instead), it preserves the efficiency thesis, and it fits the deadline with margin. Tier 2's foundation backbone belongs only as a one-head ablation; Tier 3's cross-task attention is a skip (186 Q7). If the MVP + baselines come back strong and time remains, *then* layer the headroom ablation — but the paper does not depend on it.

### Q11 — Is there a "Tier 0" cheaper than Tier 1 that still clears 70–80%?

**Yes — it's the re-scoped plan, and it is the right path.** The insight 188–190 miss: the cheapest high-value moves are **bug-fixes, not redesigns.** Concretely: verify eval (free, potentially decisive); PSR temporal-resolution + focal loss (1 day); detection TAL-assigner drop-in onto the *existing* head + move off raw-P2 features (1–2 days); activity — *do nothing*, let the 2-layer MLP train; data coverage (8000-batch cap, already E-7) + EMA (already B-4); soup init (near-free). This "Tier 0" plausibly reaches 60–80% of the **IndustReal-only** ceilings on detection/activity/pose with **PSR the honest miss**, and — crucially — it **keeps the efficiency claim intact.** That is the publishable paper, at a fraction of Tier 1's cost.

### Q12 — Which 188–190 recommendations do you disagree with?

- **Foundation backbone as headline (189 Tier 2):** disagree (Q4, FC-3). Ablation only.
- **From-scratch YOLOv8 head (188 §1.3):** disagree on scope (Q1, FC-1, FC-2). Port TAL+loss onto the existing decoupled head.
- **ArcFace + temporal-attention-pool activity head (188 §2.3/2.4):** disagree (Q3, FC-5). Trust the MLP; use logit-adjustment only if needed.
- **"STORM-like" decoder (188 §3.3):** disagree (Q2, FC-4, FC-7). Fix temporal resolution + loss first; accept PSR as the honest miss.
- **Cross-task attention:** **agree with 188/189's own hedging — skip it** (186 Q7; high risk of task-token collapse, needs re-plumbing, redundant).
- **MMoE:** **skip** (186 D-1). Marginal upside, expert-collapse risk on ~75K frames; the bottleneck isn't sharing topology.
- **MixUp / heavy augmentation:** **task-specific, not global.** For **activity** it likely *hurts* the long tail (186 Q6) — skip or make class-aware. For **detection** — which *is* data-limited (186 C-5) — YOLO-style mosaic/mixup genuinely helps. So apply augmentation to the detection branch, not to activity. Do **not** temporally jitter PSR (order *is* the label — 186 Q6).
- **Model soup:** **agree it's worth it** — but as a near-free init increment, not a 3-week tier (Q5).
- **The 3-tier escalation framing itself:** disagree — it presents "beat SOTA" as achievable and central when it is neither.

---

## 4. VALIDATION / CRITIQUE OF THE 3-TIER PLAN (189 §3)

| Tier (189 §3) | Verdict | Why |
|---|---|---|
| **Tier 1** (MViTv2-L + YOLOv8 head + STORM decoder + shared MTL, ~10d, "70–80%") | **Over-scoped and mis-targeted.** | MViTv2-L is an unnecessary 2× compute tax on a non-binding bottleneck; the from-scratch YOLOv8/STORM heads risk new bugs on the timeline. Replace with targeted fixes (§5). |
| **Tier 2** (frozen InternVideo2-L/DINOv2-L + LoRA + head upgrades, ~15d, "80%") | **Reject as headline.** | Inverts the efficiency claim (FC-3), breaks the WACV comparison, kills "faster inference," license risk. Legit only as a one-head headroom ablation. |
| **Tier 3** (Tier 2 + sequential pretrain + soup + cross-task attn, ~25d, "95%") | **Reject.** | Doesn't fit the deadline; cross-task attention is a known-risky skip (186 Q7); "95% of four specialists in one shared model" is the trap 186 named. |

**The tiers optimise the wrong axis** (absolute SOTA via capacity) and each step up *weakens* the paper's real contribution while raising cost and risk. The soup init (buried in Tier 3) and the head *bug-fixes* (misframed as Tier-1 redesigns) are the only genuinely valuable ingredients — extract them and drop the escalation.

---

## 5. THE REVISED PLAN ("Tier A" — keep MViTv2-S, diagnose, fix, prove)

**Week 1 — Diagnose + finish the headline run.**
1. **MVP smoke suite (§6), 1–2 days, on GPU-2.** Resolve which "0.0"s are eval/fresh-init artifacts. *Gate everything below on this.*
2. Let the **Path-D MTL run reach ep30–50** (it is the headline model; ep7 is not diagnostic — the act/PSR heads are cold).
3. Reproduce the **known-good PSR eval** (target the repo's 0.347) on P5 features to get a real floor.

**Week 2 — Targeted fixes (only where the MVP says they're needed) + mandatory baselines.**
4. **PSR:** fix temporal resolution (predict at native T=8 or upsample features pre-encoder, not outputs post-encoder) + focal/asymmetric loss. *No* new decoder unless MVP proves it necessary.
5. **Detection:** port **TAL assigner + DFL/CIoU/BCE loss** (cite TOOD/GFL/YOLOX) onto the **existing** decoupled head; move classification off raw-P2 onto semantic levels. Add mosaic/mixup **to detection only**.
6. **Activity:** nothing — let the 2-layer MLP + plain CE train; add logit-adjustment *only if* ST-activity shows a ceiling MTL can't reach.
7. **Run the 4 single-task baselines** (det, act-clip, PSR, pose) — required for the paper regardless. Parallelise on GPU-2.
8. **Soup the backbones** (5 min) → one MTL finetune from soup at low LR (Path-D fixes + EMA). Keep if the soup'd loss beats cold init; drop otherwise.

**Week 3 — Eval + write.**
9. Align eval protocol to WACV (clip-level activity `ACT_CLASS_GROUPING="none"`; detection **dual protocol** vs the **IndustReal-only** 0.779/0.575; no subject overlap — 186 H-1/2/3).
10. Compute per-head **MTL/ST ratios with CIs** + **efficiency** (43.5M vs ~120M specialists, one forward pass) + the **E8 gradient-cosine heatmap** (Figure 1).
11. Write the **L2 (positive transfer) + L3 (efficiency) + method (Kendall-collapse fix)** paper. Keep the negative-result/pathology section (186 J-4). Title stands (186 J-1).

**Cost:** ~10–13 GPU-days, comfortably inside 3–4 weeks. **No backbone swap, no from-scratch heads, no tier escalation.**

---

## 6. THE MINIMUM-VIABLE SMOKE TEST (Q9, expanded)

Run on GPU-2 while the Path-D run continues on GPU-1. Total ~1.5 days. **Purpose: separate "not working" from "not converged yet" and "eval is lying."**

**Probe 1 — Eval-harness sanity via overfit (½ day; do this FIRST).**
For each head, take **200 fixed images/clips**, train the head (backbone frozen) to near-zero train loss (~200–500 steps), then run the *real* eval on those same 200.
- ✅ Pass: eval metric climbs well above chance (det mAP ≥ 0.5, act top-1 ≥ 0.8, PSR F1 ≥ 0.5 on the overfit set).
- ❌ Fail (train loss →0 but eval metric stays ~0): **the eval harness / target encoding is broken.** Fix that before any architecture work. *This is the probe that most likely explains detection's 0.0 (cf. the 0.468 prior run).*

**Probe 2 — ST-activity(clip), 5 epochs (½ day).** Backbone trainable, 2-layer MLP head, plain CE, clip-level top-1.
- ≥ 0.30 by ep5 ⇒ head+backbone adequate; residual is MTL cost. **Do not touch the activity head.**
- < 0.10 by ep5 ⇒ deeper issue (data/label/eval) — investigate before ArcFace ever enters the conversation.

**Probe 3 — PSR on P5: reproduce 0.347 + temporal-resolution A/B (½ day).**
- (a) Run the known-good PSR eval on the P5 head → confirm non-zero F1 (target ~0.3).
- (b) A/B: current (interpolate outputs 8→16) vs predict-at-T=8. If predict-at-T=8 F1 ≫ interpolated, FC-4 is confirmed and the fix is a few lines — no decoder needed.

**Probe 4 — Detection TAL-lite vs 3×3 on the overfit set (½ day).** On 200 images: current 3×3 assigner vs a minimal TAL port.
- 3×3 already overfits to mAP ≥ 0.6 ⇒ assigner isn't the bottleneck (features/eval are) ⇒ don't rewrite the head.
- TAL materially higher ⇒ the ~2-day TAL port (§5.5) is justified.

**Go/No-Go:** After ~1.5 days you know, per head, whether the blocker is *eval*, *fresh-init/undertraining*, a *cheap bug*, or a *genuine architecture ceiling*. Commit expensive work **only** to the last category — which, on current evidence, is at most PSR, and even there 186 already licensed you to accept it as the honest miss.

---

## 7. NEW STRATEGIC INSIGHTS

1. **The team is pattern-matching "below SOTA ⇒ add architecture."** But 186 *proved* these SOTAs are set by small models. Below-SOTA here is dominated by fresh-init heads, a possibly-broken eval, two one-file bugs, and undertraining. Escalating architecture before ruling those out is the classic way to burn a deadline on non-problems. The MVP is the discipline that stops it.

2. **The efficiency claim is the paper's spine, and the plan would snap it.** Every "beat-SOTA" lever in 188–190 (foundation backbone, bigger heads) trades away the one claim that is *provably true regardless of accuracy* (43.5M, one forward pass, ~2.7× fewer params than the real specialists). Guard it: **the backbone stays small.**

3. **Reframe the target from "beat SOTA" to "characterise transfer."** Your differentiated, reviewer-proof contribution is not another 0.8 mAP — it's the **Kendall-collapse → inverse-loss-scaling** diagnosis + fix, plus a clean **per-task positive/negative-transfer map** on a real industrial dataset. That paper is *stronger* than a marginal SOTA chase and it's the one your data already supports.

4. **Pre-register PSR as the miss and detection's target as the IndustReal-only ceiling.** Both moves inoculate you against the reviewer objection you'd otherwise invite by quoting 0.838/0.901 as if reachable.

5. **Every expensive idea in 188–190 has a cheap precursor that must be tried first:** eval-verify before head-rewrite; feature-source before TAL; balanced-softmax before ArcFace; temporal-resolution fix before a decoder; soup-init before a foundation backbone. Do the precursors; most of the expensive ideas will prove unnecessary.

---

## 8. ONE-SCREEN SUMMARY

```
188–190 PROPOSE                         →  ROUND-3 VERDICT (code + provenance)
────────────────────────────────────────────────────────────────────────────
"Gap is representational + capacity"    →  FALSE. SOTA set by MViTv2-S / a SMALLER CNN /
                                           a small tracker. Gap = eval artifacts +
                                           fresh-init heads + 2 one-file bugs + undertraining.
Foundation backbone (Tier 2, headline)  →  REJECT. Inverts efficiency (real specialists ≈120M,
                                           not 400M), breaks WACV comparison, kills fast-inference,
                                           license risk. Ablation-only.
From-scratch YOLOv8 head                 →  OVER-SCOPED. Decoupled head ALREADY exists; no YOLOv8
                                           paper exists (cite TOOD/GFL/YOLOX). Port TAL+loss only,
                                           after verifying eval + fixing feature source.
"STORM-like" GRU decoder                 →  SKIP (unverified). Real PSR bug = T16→8→16 interpolation
                                           + rare-event loss. Fix those; accept PSR as the honest miss.
ArcFace + temporal-attention activity    →  SKIP. SOTA hit with 1 linear + plain CE. 0.008 = ep0
                                           (fresh-init). Per-frame tokens aren't even exposed.
Model soup / Strat-4 (2–3 wk tier)       →  DO THE SOUP (near-free init), but the 4 ST runs are
                                           MANDATORY BASELINES, not tier overhead. No 3-week tier.
MTL/ST ≥0.9 global gate                  →  DROP. Report ratios+CIs; positive transfer (>1) is the
                                           headline; efficiency is separate; PSR<1 is fine.
────────────────────────────────────────────────────────────────────────────
DO INSTEAD (Tier A): MVP smoke suite (1.5 d) → keep MViTv2-S → let Path-D reach ep30–50 →
  fix PSR temporal-res + loss, port TAL onto existing det head + fix det features, leave
  activity alone → run 4 ST baselines → soup init → prove L2+L3+method with honest per-head
  ratios + efficiency + the E8 gradient heatmap.  ~10–13 GPU-days. Fits the deadline.
FIRST MOVE: overfit 200 images per head and check the EVAL metric moves (the 0.468 prior run
  says detection's 0.0 may be an eval bug, not an architecture gap).
```

*Companion to 181 (round 1), 186 (round 2), 187 (status), 188–190 (the plan under review), 191 (round-3 prompt). Where this file and 188–190 disagree, this file is grounded in the executed code and the SOTA provenance, and supersedes them — same authority rule as 181 and 186. The short version: you do not need a bigger model or four new heads; you need to prove the current numbers are real, fix two bugs, run the baselines you already owe the paper, and tell the honest efficiency + positive-transfer + Kendall-fix story you already have the data for.*
