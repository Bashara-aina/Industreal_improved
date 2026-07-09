# 186 — Opus Round 2 Answer: How to Reach ≥80% of SOTA on Every Head

**Date:** 2026-07-09
**Inputs read line-by-line:** 176 (progress + SOTA provenance), 181 (Opus round-1 answer), 182 (strategic overview), 183 (architecture), 184 (training & data), 185 (50 questions).
**Ground-truth re-verified against code:** `src/models/mvit_mtl_model.py` (447 lines), `scripts/train_mtl_mvit.py` (1683 lines, post-Path-D commit `71df66759`), `eval_results.json`, and the SOTA source table in 176 §4.1.
**Authority rule (same as 181):** where the strategy docs (182–185) and the code/SOTA-provenance disagree, the code and the provenance win, and I say so explicitly.

---

## 0. THE ONE-PARAGRAPH ANSWER (and it corrects 182)

**Do not do Strat-2 as the headline.** File 182 recommends replacing MViTv2-S with a frozen 300M–1B foundation model because "MViTv2-S won't scale to 80% SOTA" (183 §1.3). **That premise is false, and the primary source proves it.** I extracted and read the IndustReal paper's benchmark tables directly (Schoonbeek et al., WACV 2024, arXiv:2310.17323): **Table 2 states verbatim that "the SlowFast CNN and MViTv2-S transformer are chosen to benchmark" action recognition, and MViTv2-S (Kinetics-pretrained, RGB) scores 65.25% top-1 / 87.93% top-5** — this *is* the activity SOTA you are chasing, set with your exact backbone class (the multimodal RGB+VL+stereo *ensemble* reaches only 66.45%). **Table 3 shows detection SOTA of 0.838 mAP is a YOLOv8-m** — a ~25M CNN, *smaller* than your 34.5M backbone — and that 0.838 requires the **IndustReal+synthetic** training scheme (IndustReal-only YOLOv8m = 0.779 b-boxed / 0.575 entire-video). PSR SOTA (0.901, per 176) is **STORM**, a separate purpose-built state-tracking method (it is not even in the original IndustReal paper), not a bigger backbone. So the gap to 80% is **not representational capacity**; it is three separable, already-diagnosed problems: (1) **activity is starved by the Kendall weighting** (weight 0.04 — Opus 181's core finding, now fixed by Path-D), (2) **detection has the wrong head/assignment** (a hand-rolled center-assignment on temporal-pooled transformer features vs. a dedicated YOLOv8 neck), and (3) **PSR reads `conv_proj` — literally the patch-embedding, layer 0** — which carries no semantics. The right program is **keep MViTv2-S, let the Path-D run finish, and fix the three heads in place (Strat-1++ merging into Strat-4)**, running the three baselines from 181 in parallel. Strat-2 should be **demoted to an optional ablation**, because it (i) is unnecessary to clear the bar, (ii) *breaks* the one-shared-backbone comparison to the WACV baseline that makes your MTL claim legible, (iii) *contradicts your own efficiency thesis* by making the model 10–30× larger, and (iv) carries real license/infra risk (InternVideo2 weights are **not** cleanly Apache-2.0 — see §5). **And the deepest strategic correction of all:** "≥80% of SOTA on *every* head, simultaneously, in *one* 40M shared-backbone model" is a *harder* problem than any single one of those SOTAs — each was set by a *different specialist* (YOLOv8m, STORM, MViTv2-S). Requiring one shared model to beat 80% of four specialists at once is not the publishable bar. The publishable bar is Opus 181's **L2 (positive transfer on ≥1 head) + L3 (efficiency) + the methodological Kendall-pathology fix**, with per-head numbers reported honestly against each specialist.

---

## 1. GROUND-TRUTH VALIDATION (what I verified in the code, this round)

| Claim in 182/183/184 | Verified? | Evidence (file:line) |
|---|---|---|
| Backbone is MViTv2-S, 34.5M, Kinetics-400, **fully trainable (not frozen)** | ✅ | `mvit_mtl_model.py:48`, `176 §1` "all backbone params trainable" |
| Activity head = `LayerNorm(768) → Linear(75)`, on the **single pooled class token** | ✅ | `mvit_mtl_model.py:219-236, 414` — head sees `cls_token=[B,768]`, **not** `[B,T,768]` |
| PSR head reads `conv_proj` (the patch-embed / P2), 96-ch, 3-layer causal transformer, d=96 | ✅ | `mvit_mtl_model.py:243-302, 417` — `psr_input = fpn_feats.get("P2")` |
| Detection = FPN over hooked stages + decoupled cls/box(DFL) head, 24 classes | ✅ | `mvit_mtl_model.py:129-212, 405-411` |
| Path-D **is implemented** (EMA-norm, class-wt sqrt-tame, log_var caps, grad-accum, 3×3 pos cells, α=0.5) | ✅ | commit `71df66759`; `train_mtl_mvit.py:703-747` (EMA+caps), `:333-370` (sqrt weights, LS 0.05), `:247-264` (3×3), `:152` (α=0.5), `:812-818` (accum) |
| SOTA: act 0.6525 (**MViTv2-S**, Kinetics, RGB), det 0.838 (YOLOv8-m, b-boxed) / 0.641 (entire-vid), PSR 0.901 (STORM), pose = **no SOTA** | ✅ **verified in the paper's Table 2 & 3** | arXiv:2310.17323, extracted & read this round |
| "MViTv2-S too weak / needs foundation model" (183 §1.3) | ❌ **REFUTED by the primary source** | Table 2: activity SOTA *is* MViTv2-S; Table 3: det SOTA is a *smaller* CNN (YOLOv8-m), and needs synthetic-data pretraining to reach 0.838 |
| grad-accum fix makes effective batch = 4 (a clean mean) | ⚠️ **partial** | `:798/809` backward does **not** divide by `grad_accum_steps` → grads are *summed*, not averaged (see §5.1) |
| 3×3 positive-cell fix gives valid box supervision on all 9 cells | ⚠️ **latent bug** | `:260-263` DFL/IoU targets use **GT-center-relative** offsets for *every* cell in the patch, ignoring the cell's own offset (see §5.2) |

The one thing I could **not** verify from this ephemeral container is the *live* Path-D run's post-ep6 metrics (the GPU run is on your machine, not here). Everything else is now confirmed against the primary source: I downloaded arXiv:2310.17323, extracted the text, and read Tables 2 and 3 directly. The activity=MViTv2-S / 65.25% and detection=YOLOv8-m / 0.838 (b-boxed) attributions are **verbatim from the paper**, not just the repo's 176 §4.1. The strategic implication — *this dataset's activity SOTA is reachable with the MViTv2-S backbone, and its detection SOTA with a smaller CNN* — is therefore not an inference; it is what the benchmark paper reports.

**One provenance flag on the "PSR F1 = 0.0" claim:** the repo's `eval_results.json` shows `psr_overall_f1 = 0.3469` / `psr_component_f1 = 0.4409` at an early checkpoint of a *different* pipeline. So PSR is **not intrinsically pinned at 0**; the "0.0" is specific to the MTL run's `conv_proj` feature source + weight starvation. This strengthens the "fix the feature source" diagnosis and weakens any "PSR is fundamentally impossible" fatalism.

---

## 2. DIRECT ANSWERS TO THE 10 OPEN QUESTIONS IN 182 §8

**Q1 — Is frozen-backbone+adapter (Strat-2) more defensible than single-task-pretrain→MTL-finetune (Strat-4) for AAIML?**
**No — Strat-4 is far more defensible, and Strat-2 actively hurts this paper.** Your thesis is *MTL efficiency + positive transfer* on IndustReal. Strat-2 grafts a 300M–1B pretrained model on top; a reviewer's immediate, correct objection is *"of course a video foundation model lifts every head — that says nothing about your MTL contribution, and it destroys your efficiency claim (40M → 300M+)."* It also breaks apples-to-apples comparison with the WACV MViTv2-S baseline. Strat-4 keeps the 40M shared backbone, so every number is directly comparable to the specialist baselines, and its ingredients (Kendall-pathology fix, task-arithmetic/model-soup init, cross-task-transfer analysis) are genuinely novel. **Verdict: Strat-4 (or the cheaper Strat-1++ that converges to it), MViTv2-S retained. Strat-2 = optional "scaling" ablation only.**

**Q2 — Given 1 primary GPU: Strat-4 sequentially (~10 d) or Strat-2 + 4 ST in parallel (~3 wk)?**
Wrong trade to agonize over, because Strat-2 shouldn't be on the critical path. The real plan costs far less: (a) the Path-D MTL run is **already launched** (~22 min/epoch → ep30 in ~11 h); (b) 181's three artifacts — ST-activity(clip), ST-detection(with assignment fix), and the 2-hour gradient-flow diagnostic — are ~2–3 GPU-days total and can run on the 2nd GPU. **That is a complete, publishable evidence set in ~1 week on your hardware.** Reserve Strat-4's per-task pretraining + soup + finetune (~5–6 GPU-days) as the "push the numbers up" phase *after* you see ep30. Don't spend the GPU on foundation-model plumbing.

**Q3 — Is a 2-layer MLP head *ever* sufficient for 75-class video activity, or do we need a temporal transformer regardless of backbone?**
**A 2-layer MLP is almost certainly sufficient — and 183 overstates the head-capacity problem.** Proof: the WACV activity SOTA (0.652) was reached with MViTv2-S whose native classifier is *a single linear layer on the pooled class token* — exactly your current head. The MViT class token already performs learned spatiotemporal attention pooling over all 16 frames; it is **not** a mean-pool that a temporal head must repair. Your activity failure is dominated by (i) the Kendall weight 0.04 starvation (now Path-D-fixed) and (ii) undertraining/data-coverage, **not** by head width. **Caveat that matters:** the current forward pass only exposes the *pooled* `cls_token=[B,768]` to the head (`mvit_mtl_model.py:414`); it does **not** expose per-frame tokens `[B,T,768]`. So the fancy "AttentionPool over T" head in 183 §2.1 **cannot even be built without re-plumbing the backbone forward** — another reason it's the wrong first move. Recommendation: add a cheap `Linear(768→768)→GELU→Linear(768→75)` (2-layer MLP) as insurance; do **not** invest in a temporal transformer head. If ST-activity(clip) on the frozen class token clears ~0.45–0.55, the head is proven adequate and the residual gap is pure MTL cost.

**Q4 — Does PSR fundamentally need block-3+ features, or can block-1/`conv_proj` work with long temporal context (T=64)?**
**It fundamentally needs deeper features; long T over shallow features will not save it.** `conv_proj` is the strided patch-embedding — layer 0. It is a linear projection of raw pixels; there is no object/state semantics for a transformer to bind transition events to. Stacking a 64-frame temporal transformer on semantics-free features gives you a longer sequence of noise. Move the PSR source to **`blocks[14]` (P5, 768-ch, post-all-attention)** or a mid-block (`blocks[3]`, P4) — the hooks already exist (`mvit_mtl_model.py:81`), so this is a **one-line pointer change** plus adjusting the head's input dim. Longer temporal context is a *secondary* gain **after** the feature source is fixed, and note STORM (the 0.901 SOTA) is a dedicated state-tracking decoder — matching 0.72 (80%) with a lightweight head on shared features is your **single hardest bar**; treat PSR as the rate-limiter and the most likely honest limitation.

**Q5 — Is YOLOv8's TaskAlignedAssigner the right drop-in, or SimOTA/OTA?**
**TAL is the right choice; do not detour into SimOTA/OTA** (they solve the same dense-assignment problem with more moving parts and no benefit here). But two caveats bound the upside: (a) your detection features are **temporal-pooled MViT stages** at strides 4/8/16/32 (`mvit_mtl_model.py:410`), not a YOLOv8 CSP/PAN neck — a shared video backbone will likely *not* fully match a specialist YOLOv8m's localization, so 0.67 (80%) is *plausible but not guaranteed*; (b) before crediting *any* box-quality gain, fix the DFL-target bug in the current 3×3 code (§5.2) — otherwise TAL inherits mis-placed regression targets. Frame detection for the paper as **"shared-backbone MTL reaches X% of the YOLOv8m specialist,"** not "we match 0.838."

**Q6 — Is MixUp+CutMix+RandAugment enough augmentation, or do we need domain-specific augmentation?**
Augmentation is the **third-order** lever here, behind (1) MTL weighting and (2) data coverage — don't over-invest. Specifically: **MixUp/CutMix will likely *hurt* the 75-class long tail** (blending dilutes the rare-class signal, and label-mixing on procedural states is semantically dubious); if used, make it *class-aware* or skip it for activity. **RandAugment is image-designed** — use a light setting and apply spatial ops *consistently across the 16 frames* (the code path in 184 §3.1 does this). The highest-value "augmentation-adjacent" fix is not augmentation at all: it's raising the **4000-batch cap** (currently `max_batches_per_epoch`, `train_mtl_mvit.py:1512`), which limits each epoch to ~10% of the ~78k windows. Domain-specific temporal jitter is fine but secondary; **do not temporally jitter PSR** (temporal order *is* its label).

**Q7 — Does cross-task attention help when each head already has an adapter, or is it redundant?**
In a **frozen-backbone+adapter (Strat-2)** setup it is largely **redundant and destabilizing** — each head already gets task-specialized features from its adapter, and cross-task attention adds task-token-collapse failure modes (185 D-5). In a **shared-trainable-backbone (Strat-1/4)** setup it *could* help, but it's **high-risk/medium-reward and premature**: your current heads don't even receive per-token features (activity/pose get the pooled token; detection gets FPN maps), so cross-task attention requires re-plumbing the forward pass. **Verdict: skip cross-task attention for the first paper.** If you want a topology upgrade, MMoE is cheaper and safer, but even that is optional given the bottleneck is heads/weighting, not sharing.

**Q8 — Is "MTL via foundation-model adapters" a strong enough AAIML narrative, or do we need the task-arithmetic + cross-task-attention story?**
"MTL via foundation-model adapters" is the **weaker** narrative for *this* paper, not the stronger one (see Q1). The strongest, most reviewer-proof narrative is the one Opus 181 already identified and you already have data for: **"Uncertainty-weighted MTL silently degenerates to inverse-loss scaling and starves the highest-loss head; we characterize it, correct it (Path-D), and show one 40M shared-backbone model performs 4 assembly tasks in a single pass, with positive transfer to head-pose, at 2.3–3× fewer params than 4 specialists."** Task-arithmetic/model-soup init is a clean, cheap addition to that story. You do **not** need cross-task attention.

**Q9 — If Strat-2's adapters fail on activity specifically, is there still a paper?**
Yes — and this is another reason to not gate the paper on Strat-2 or on absolute 80%. Per 181 §5 Q3, the defensible contribution is **L2+L3+method**, not "beat 80% on all four." If activity lands at, say, 0.35 in MTL while ST-activity(clip) shows a 0.55 ceiling, you have *quantified MTL's cost on activity* (a legitimate finding) while still claiming positive transfer on pose and efficiency overall. The paper survives a soft head; it does **not** survive making the model 300M and calling it efficient.

**Q10 — Given 2026-07-09 and a Sept/Oct AAIML deadline, is 3–4 weeks realistic, or ship Strat-1 only?**
With the reframed (keep-MViTv2-S) plan, **you are comfortably inside the deadline** — the critical path is ~1 week to a full evidence set (Path-D ep30 + 3 baselines), then ~1 week of targeted head fixes (PSR feature source, detection DFL-target fix + TAL, optional 2-layer act head), then ~1 week of writing. The *only* thing that would blow the timeline is exactly the thing 182 recommends: standing up a foundation-model backbone, verifying licenses, and re-tuning at 3–10× the compute per epoch. **Ship the corrected Strat-1++/Strat-4-lite, not Strat-2.**

---

## 3. THE 4 DECISION-DRIVING QUESTIONS OF 185 (§ "FINAL")

**Q-Backbone (185 B-1/B-2/B-3): which backbone closes the gap?**
**Keep MViTv2-S.** The gap is not the backbone (activity SOTA = MViTv2-S; detection SOTA = a *smaller* CNN). Scaling to MViTv2-L (185 B-3) is a mild, license-clean hedge if you have spare compute, but it addresses a bottleneck that the evidence says isn't the binding one. Foundation models (B-1/B-2) are an *ablation to show headroom*, not the headline.

**Q-Architecture (185 C-1/C-3/D-3): head capacity vs. topology vs. features — highest leverage?**
Ranked by validated leverage: **(1) MTL loss weighting** (Path-D, already the biggest single fix — turns activity weight 0.04→≥0.37), **(2) PSR feature source** (`conv_proj`→`blocks[14]`, one line, highest per-task upside for the worst head), **(3) detection assignment+target correctness** (TAL + fix the DFL-target bug), **(4) model-soup/task-arithmetic init** (D-3, near-free, good init for MTL finetune). Head *capacity* (C-1) and *topology* (MMoE/cross-task, D-1/D-2) are **low leverage** here and should not be prioritized.

**Q-Strategy (185 A-1/A-4/G-3): is "80% of SOTA" the bar, or "80% of single-task ceiling"?**
**"80% of single-task ceiling" is the right, honest, publishable bar; "80% of four different specialists' SOTA simultaneously in one shared model" is a self-imposed trap.** Reason: the four SOTAs are set by four *different* purpose-built systems. Adopt the L2/L3+method framing (181). Report each head against *both* the specialist SOTA (for context) and a *matched single-task MViTv2-S/YOLOv8 baseline you train* (for the real MTL-cost claim). This is the single most important strategic decision in the whole 182–185 stack, and 182 gets it wrong by anchoring on absolute SOTA.

**Q-Compute (185 F-1/F-4): can Strat-4 fit in 2 weeks on 1 primary GPU?**
Yes for the *corrected* plan (§2 Q2/Q10). Use the 2nd GPU for the ST baselines in parallel (185 A-8/F-1). Raising the batch cap 4000→8000 (F-4) roughly doubles per-epoch wall-clock for 2× data coverage — worth it once, not for every ablation.

---

## 4. THE 50 QUESTIONS OF 185 — VALIDATED VERDICTS (grouped; decision-relevant ones first)

**Section A (strategy):** A-1 → optimize the *comparison* (MTL vs matched ST), not absolute SOTA. A-2/A-10 → confirm the actual AAIML deadline with the user (unknown in-repo; the plan fits Sept/Oct regardless). A-3 → keep 4-ST scripts as the fallback, but you won't need Strat-5. A-4 → yes, "MTL reaches X% of ST ceiling at 2.3–3× fewer params" is an accepted AAIML-style contribution. A-5/A-7 → **do not add a 5th task**; lead with the MTL-model + pathology-fix, keep the pathology section (it's your reviewer shield). A-6 → pose has **no SOTA** (176 §4.1), so "≤12°" is arbitrary; report absolute MAE and stop treating pose as a "bar." A-8 → use the 2nd GPU for parallel baselines.

**Section B (backbone):** B-1/B-2 → moot given "keep MViTv2-S"; if you *ablate*, EVA-02 (image) vs InternVideo2 (video) is a real toss-up and should be decided by a 1-epoch smoke test, not prose. B-3 → MViTv2-L is the *only* backbone change I'd even consider, and only as a hedge. B-4/I-1/I-5 → video-language/zero-shot for activity is a **wildcard, not a plan**: your class labels are state-codes ("10000000000"), not CLIP-friendly text, so zero-shot needs hand-written text descriptions first — interesting future work, not this paper. B-5 → YOLOv8m as a *separate* detector defeats the shared-backbone thesis; it's the **specialist baseline**, not the MTL model. **B-6 → highest-value cheap experiment in the whole list: a 1-epoch PSR ablation over `conv_proj` vs `blocks[3]` vs `blocks[14]` features.** B-7 → frozen-image-backbone + temporal head is Strat-2-flavored; ablation only. B-8 → no public assembly-pretrained backbone; skip.

**Section C (head capacity):** C-1 → 2-layer MLP suffices (see §2 Q3). C-2 → ArcFace is a *maybe +2–5%* on the long tail; try it **only** after Path-D + ST-activity establish the ceiling — not a first move. C-3 → block-14 features + deeper head is the PSR path; T=32 is secondary. C-4 → TAL should converge <30 epochs *if* the DFL-target bug is fixed. C-5 → 24-class assembly with ~6k labeled frames is *data-limited*, so detection augmentation (mosaic/mixup à la YOLOv8) matters more here than elsewhere. C-6 → geodesic pose loss: skip, pose works. C-7/C-8 → no 5th head; per-task adapters only *if* you ever do Strat-2.

**Section D (topology):** D-1 (MMoE) / D-2 (cross-task attn) → **low priority**, bottleneck is elsewhere; D-2 also needs forward-pass re-plumbing. **D-3 (model soup) → yes, do it — near-free strong init for the MTL finetune** (Wortsman 2022 / Ilharco 2022 task arithmetic). D-4 → yes, lower LR (5e-5 backbone / 5e-4 heads) for the finetune. D-5 → cross-task tokens collapse; another reason to skip. D-7 → share LN across tasks (fine). **D-8 (is PCGrad still needed post-Path-D?) → answer empirically with the E8 gradient-flow diagnostic; if it's a no-op, drop it for ~30% throughput** (181 K-5).

**Section E (recipe):** E-1 → MixUp likely hurts long-tail activity; class-aware or skip. E-2 → RandAugment light + frame-consistent. E-3/I-8 → **add EMA/SWA model weights — cheap, reliable +1–2%**. E-4 (Lion) → only relevant if you ever train a huge backbone; skip for MViTv2-S. E-5 → 10-epoch warmup only matters for frozen-backbone (Strat-2); N/A here. E-6 → **try grad-clip 1.0→5.0**: with the summed-grad accumulation (§5.1) a clip of 1.0 is likely over-clipping. **E-7 → raise the 4000-batch cap to 8000 once** (biggest "free" convergence lever). E-8 → cosine warm-restart: cheap, optional. E-9/G-6 (bf16 NaN) → watch ep1–2; the code already has finite-guards (`:750`). E-10 → PSR label smoothing: minor, optional.

**Section F (compute):** covered in §3 Q-Compute. F-2/F-5/F-6 → only relevant to the Strat-2 ablation; measure batch size empirically if you run it.

**Section G (risk):** G-1/G-3 → paper survives a soft head via L2/L3+method (§2 Q9). **G-2 → PSR (0.72) is the realistic 80%-miss** — plan the narrative around it now. G-4 → license: **Apache-2.0 does not automatically cover InternVideo2's *weights*** (§5.3); another Strat-2 caution. G-5 → if Path-D ep30 doesn't lift activity off the floor, the problem is *coverage/epochs or a residual bug* (check §5.1), not the backbone.

**Section H (metrics — do these before quoting any SOTA comparison):** **H-1/H-2/H-3 are non-negotiable:** confirm your eval protocol matches WACV (activity clip-level top-1 with `ACT_CLASS_GROUPING="none"` per 176 §4.2; detection **dual protocol** annotated-frames↔0.838 *and* full-video↔0.641; **no subject overlap** across train/val/test — 176 says test=10 subjects, val=5). A metric mismatch here invalidates every comparison. H-4 → pose bar is arbitrary (no SOTA). H-5 → report ratios in the main table, deltas in supplement. H-6 → "positive transfer" = MTL ≥ matched-ST on the same backbone (needs the baselines).

**Section I/J (wildcards & narrative):** I-2 (track-by-detect activity) depends on detection working → not now. I-3 (self-sup pretraining on unlabeled assembly video) → nice future work if unlabeled video exists. I-9 (4-ST ensemble) → that's the *specialist* comparison, report it. **J-1 title:** *"One Backbone, Four Tasks: Diagnosing and Fixing Uncertainty-Weighting Collapse in Multi-Task Assembly Understanding."* **J-2 headline figure:** the per-task gradient-cosine heatmap (E8) beside the MTL-vs-ST bar chart. **J-3 hero result:** pose positive transfer (clean) + the activity *recovery* curve after the Kendall fix (dramatic). **J-4:** yes, keep the negative-result/pathology section — it's your most defensible contribution.

---

## 5. CONCRETE CODE-LEVEL FINDINGS (two residual defects in the Path-D "fix")

These are new, code-grounded, and directly affect whether Path-D delivers. Neither is in 182–185.

### 5.1 Grad-accumulation now accumulates, but **sums instead of averages** (2× effective LR at the boundary)
`train_mtl_mvit.py:798/809` does `scaler.scale(total_loss).backward()` on **every** micro-batch, and the optimizer steps once per `grad_accum_steps=2` boundary (`:812-818`). Opus 181 D4 correctly moved `zero_grad` to after `step()` — so accumulation is no longer a no-op — **but the recommended `total_loss / grad_accum_steps` scaling was not applied.** Result: the two micro-batch gradients are **summed**, so the boundary step sees ~2× the intended magnitude (an effective 2× LR), which `grad_clip_norm=1.0` then largely clips — coupling the accumulation to the clip in a way that makes both hard to reason about. **Fix:** `scaler.scale(total_loss / grad_accum_steps).backward()` and divide the PCGrad backbone grads by the same factor before accumulating (`:802-807`). This also explains why grad-clip 5.0 (185 E-6) may help: at 1.0 you're clipping summed grads.

### 5.2 The 3×3 positive-cell fix uses **GT-center-relative** DFL/IoU targets for *all* 9 cells
`train_mtl_mvit.py:260-263` sets, for every cell in the 3×3 patch, `dfl_target = (gt_cx - box_x0)/stride, …` — i.e. the distances from the **GT box center** to its edges, **identical for all 9 cells**. Correct DFL/anchor-point regression targets are distances from **each cell's own center** to the box edges (they differ per cell). So the 8 off-center positives are trained to regress boxes as if they sat at the GT center → systematically biased localization, capping mAP even after the density fix helps classification. **Fix:** compute per-cell offsets `(cell_cx[ci] - box_x0)/stride`, etc. This is a prerequisite before TAL (Q5), and before quoting any MTL/ST detection mAP.

### 5.3 License caution for the Strat-2 ablation (if you run it)
DINOv2 code+weights are Apache-2.0 (relicensed 2024) — safe. EVA-02 weights are MIT — safe. **InternVideo2's *code* is Apache-2.0 but its larger *checkpoints* are gated/usage-restricted** (the HF model cards carry additional terms; the 1B/6B weights are not a clean Apache-2.0 grant). 182 §3 lists InternVideo2 as "Apache 2.0" without this distinction. If you ever do the scaling ablation, prefer **DINOv2-L or EVA-02-L** for a clean license, and verify the specific checkpoint's card — don't assume the repo license covers the weights.

---

## 6. THE PLAN (revised, ordered, keep-MViTv2-S)

**Week 1 — evidence set (mostly already running):**
1. Let the **Path-D MTL run reach ep30** (~11 h) — this is the headline model's first read.
2. On GPU-2, run 181's three artifacts: **ST-activity(clip-level)**, **ST-detection(with the §5.2 target fix)**, and the **2-hour gradient-flow diagnostic (E8)** → this is Figure 1 and settles D-8 (drop PCGrad?).
3. Fix §5.1 (grad-accum mean) and §5.2 (per-cell DFL targets) — both are correctness fixes, cheap, and gate the detection numbers.
4. Run the **1-epoch PSR feature-source ablation** (B-6): `conv_proj` vs `blocks[3]` vs `blocks[14]`. Adopt the winner (one-line change).

**Week 2 — targeted head fixes + optional soup:**
5. Move PSR to the best feature source + a slightly deeper head; add TAL to detection; add the 2-layer activity MLP as insurance; add EMA/SWA weights (E-3).
6. If ep30 shows activity/PSR still short: run **Strat-4-lite** — per-task pretrain (act, det, psr), **model-soup the backbone** (D-3), MTL-finetune at low LR (D-4). This is where "keep-MViTv2-S" earns the 80%-of-single-task numbers.
7. *(Optional, GPU-permitting)* the Strat-2 **scaling ablation** (DINOv2-L or EVA-02-L, one head) — to *bound headroom*, reported as an ablation, never the headline.

**Week 3 — eval + write:** align eval protocol (H-1/2/3), compute MTL/ST ratios and MTL/SOTA context, write the pathology + positive-transfer + efficiency paper.

---

## 7. BOTTOM LINE (what changed from 182)

- **182's headline (Strat-2, foundation model) is demoted to an optional ablation.** Its premise ("MViTv2-S too weak") is refuted by your own SOTA table.
- **The bottleneck is heads + MTL weighting + PSR feature source — not the backbone.** All three are cheap, in-place fixes on MViTv2-S.
- **The bar "≥80% of SOTA on every head" is the wrong target;** the right, publishable target is **L2 (positive transfer) + L3 (efficiency) + the Kendall-pathology method fix**, per-head reported honestly (PSR is the likely honest miss).
- **Two residual code defects (§5.1, §5.2) must be fixed before any detection number or "faster training" claim is credible.**
- **This all fits a Sept/Oct deadline on your existing 1–2 GPUs.** Strat-2 is the only thing here that wouldn't.

*Companion to 181 (round-1 answer) and 182–185. Where this file and 182/183 disagree, this file is grounded in the executed code and the SOTA provenance in 176, and supersedes them on the backbone decision.*
