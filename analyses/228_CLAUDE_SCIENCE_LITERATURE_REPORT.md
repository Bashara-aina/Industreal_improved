# Doc 228 — Claude Science Literature Report: MTL Methods That Close the MTL→ST Gap

**Prepared in response to doc 215's 50 deep-research questions.**
Companion artifact: `mtl_methods_evidence_table.csv` (52 methods; 43 not in doc 213).

---

## 0. Executive summary — read this first

**The single most important finding, stated honestly:** across the entire published
multi-task-optimization and multi-task-architecture literature, **no method robustly reaches
MTL/ST ≥ 1.0 simultaneously across a task set as heterogeneous as IndustReal's**
(fine-grained detection + 75-class long-tail activity + sparse binary procedural-state +
dense 6D pose), and **none has ever been evaluated on egocentric video MTL at all.** Every
"MTL beats single-task on 3+ tasks" result in the literature — and several genuinely exceed
the "1–5% transfer" band you reached in doc 213 — comes from the **NYUD-v2 / PASCAL-Context
dense-prediction regime**, whose tasks (semseg, depth, surface-normals, saliency, parsing) are
geometrically correlated, spatially dense, and gradient-commensurate. That is the opposite of
your regime. This is not a gap in our search; it is a gap in the field, and it is *your paper's
white space.*

**What that means for the four deliverable questions:**

1. **Is the central hypothesis (one MTL model matches ST on all four heads, ~2× more efficient)
   supported by prior art?** Partially. The *efficiency* half is well-supported and is the
   defensible headline. The *"matches ST on all four heads"* half is **not** demonstrated in any
   prior work for heterogeneous/video tasks — you would be the first, which is a stronger paper
   framing than "we reproduced a known result."

2. **Which published methods are worth adopting on your 30-day, time-bound budget?** In priority
   order: **FAMO** (O(1) single-backward-pass loss weighting — the only gradient-balancing SOTA
   that does not multiply your 7-day run by k=4); **head-only long-tail rebalancing**
   (Balanced-Softmax / decoupled-cRT) for activity; **MS-TCN-style temporal refinement** for PSR's
   per-frame→event-F1 gap; **MetaBalance** for the 312× activity-vs-PSR gradient-magnitude
   domination; and **Recon** (offline conflict-layer splitting) as a TIME-friendly architecture
   edit. Everything requiring per-task gradients (FairGrad, Nash-MTL, Aligned-MTL, CAGrad) is
   deprioritized purely on wall-clock.

3. **Which architecture ideas attack Kendall-collapse at the source?** **MoE / low-rank experts**
   (M³ViT, Mod-Squad, MLoRE) — the only architecture family shown to convert *negative* transfer
   into *positive* transfer — and **explicit cheap cross-task feature exchange** (MTMamba++'s
   cross-task Mamba block). Both give starved heads dedicated capacity instead of letting the
   high-loss task overwrite shared parameters. Caveat: MoE routing means near-single-pass, not
   literally one pathway.

4. **What must the paper defend against?** Three 2022–2024 papers (**Kurin NeurIPS 2022, Xin 2022,
   Elich GCPR 2024**) argue that (a) tuned sum-of-losses + standard regularization matches fancy
   MTL optimizers, and (b) the "gradient conflict causes MTL failure" narrative is weaker than
   PCGrad-era work assumed. Since your central contribution *is* a gradient-pathology diagnosis,
   you must pre-empt these by benchmarking capped-Kendall against a **properly regularized
   scalarization baseline** and against **FAMO** at matched wall-clock.

**Prioritization by doc 215 section (as requested — A & B highest):**
Section A (architecture) and Section B (optimization) carry the most new, citable, quantitative
evidence and are answered in most depth below. Sections C/D/E (per-task) yield concrete
head-level levers. Section F (efficiency) yields the reporting protocol you should adopt. Section
H (foundation models / SSMs) is mostly a "raises ceiling, not gap" negative result plus one
important foil (frozen DINOv2).

> **Provenance note.** This report consolidates four parallel literature tracks. Tracks 1–3
> (architecture, optimization, per-task) extracted quantitative Δm%/AP numbers that reproduce the
> cited papers' own tables. Track 4 (video foundation models / SSMs) is flagged by its author as
> **domain-knowledge recollection pending primary-source verification** — treat every numeric K400
> top-1 / parameter count in Section H as an unconfirmed figure to check against the paper before
> it enters your manuscript. All arXiv IDs / DOIs in the evidence table should be verified at
> citation time regardless.

---

## Section-A / Section-B priority map (per doc 215)

| doc-215 Q | Topic | Headline answer | Key new methods |
|---|---|---|---|
| **A1** | Task-conditional modulation (FiLM/adapters/gating) | Modulation works; MoE/gating is the only family that flips negative->positive transfer | M3ViT, Mod-Squad, MLoRE, Polyhistor, VMT-Adapter |
| A2 | Positive transfer on Taskonomy/NYUv2/PASCAL | Real, up to +4.82% dm, but only on correlated dense tasks | MTMamba++, TaskPrompter |
| A3 | Backbone comparison | Ceiling != gap; stronger trunk doesn't fix competition | (Section H) |
| A4 | Feature routing | Separate cls-token sources per head; explicit cross-task exchange | MTMamba++ CTM block |
| A5 | Detection necks in MTL | Grid/anchor-free shares losslessly; two-stage collapses | YOLOP |
| A7 | MTL>ST "holy grail" | Not achieved for heterogeneous/video tasks anywhere | - |
| A8 | Shared/task-specific ratio | No universal optimum; scales with task *conflict* | Polyhistor (<1-10%), MoE (more for conflict) |
| **B1** | Alternatives to Kendall+caps | FAMO (O(1)); IMTL-L; MetaBalance; DB-MTL | FAMO, MetaBalance |
| **B2** | Gradient surgery beyond PCGrad | FairGrad/Nash/Aligned-MTL strongest but k passes; Recon TIME-friendly | FairGrad, Aligned-MTL, Recon, SDMGrad |
| B3 | MTL optimizers / MGDA / FAMO | FAMO matches Nash-MTL at 1 backward pass | FAMO |
| B5 | Beating ST on all tasks | Only on NYUv2 3-task (FairGrad); never on 4 heterogeneous | FairGrad |
| B6 | Meta-learning weighting | Auto-Lambda, MetaBalance | MetaBalance |
| B7 | 2025-2026 SOTA | ConICGrad/UPGrad/PAMM/DB-MTL - frontier, unreplicated | (frontier) |

---

## Section A - MTL Architecture (answers A1-A8)

**A1 - Task-conditional modulation / FiLM / adapters / gating.** The strongest architectural
lever against your diagnosed Kendall-collapse is **Mixture-of-Experts routing**, because it is
the only design family with *published evidence of converting negative transfer into positive
transfer*. **M3ViT (NeurIPS 2022, arXiv:2210.14793)** inserts task-gated MoE layers into a ViT
and moves a plain multi-task ViT from **-6.27% -> +1.59% dm on NYUD-v2**, reaching **+2.71% on
PASCAL-Context** - a direct demonstration of reversing the exact failure mode you named.
**Mod-Squad (CVPR 2023)** adds MoE to both attention and MLP with an expert-task specialization
loss, reports **~+5.6% dt on PASCAL-Context beating all baselines on all tasks**, and prunes to
the top ~40% of experts with no loss - showing experts self-organize per task, precisely the
"dedicated capacity so activity/PSR aren't overwritten" property you need. **MLoRE (CVPR 2024,
arXiv:2403.17749)** is the parameter-efficient version: a **mixture of low-rank experts** plus a
generic all-task path adds per-task capacity at near-flat parameter cost - the most budget-aligned
way to give starved heads their own capacity. Lighter-weight modulation via **adapters/hyper-
networks** (Polyhistor NeurIPS 2022 arXiv:2210.03265: +2.34% at 6.41M params, or +1.74% at
**0.41M**; VMT-Adapter AAAI 2024: O(1) cost in #tasks) achieves small positive transfer while
pushing the task-specific parameter fraction below 10%. **Trade-off you must state:** task-gated
MoE activates a per-task expert pathway - the shared encoder runs once but expert MLPs differ per
task, so it is *near*-single-pass, not literally one pathway. If the "single forward pass / ~4x
latency" claim is load-bearing, prefer decoder-shared or low-rank-expert designs (MLoRE) over hard
task-gating.

**A2 - Positive transfer on the standard dense benchmarks.** It is now routine to beat single-task
on 3+ tasks *on NYUD-v2 / PASCAL-Context*. The cleanest all-task win is **MTMamba++ (TPAMI 2025,
arXiv:2408.15101)**: on NYUD-v2 (Swin-L) Semseg 57.01 / Depth 0.4818 / Normal 18.27 / Boundary
79.40 vs single-task 54.32 / 0.5166 / 19.21 / 77.30 - **dm = +4.82%, beating single-task on all
four tasks** (delta recomputes exactly), at lower memory and FLOPs than attention-based cross-task
decoders. **TaskPrompter (ICLR 2023, arXiv:2304.13886)** reports ~+4.49% dm on NYUD-v2. These
genuinely exceed doc 213's "1-5%, cluster 0.95-1.08" framing - **but the excess comes from (a)
larger backbones (ViT-L/Swin-L) and (b) cooperative dense tasks**, and per-task deltas remain <2
points on any single metric. **Do not quote these as evidence for IndustReal** without the caveat
that they arise in a fundamentally more cooperative regime.

**A3 - Backbone comparison for shared representations.** Covered in Section H. A stronger trunk
(VideoMAE-V2/UMT/InternVideo2) raises each head's *ceiling* by 5-11 K400 points but there is **no
evidence** any of them reduces inter-task gradient competition. Ceiling != gap. This is a useful
argument *for* your paper: it frames Kendall-collapse as an optimization/geometry problem a bigger
encoder does not solve.

**A4 - Feature routing between tasks.** Two published mechanisms transfer directly. (i)
**Decoder-focused cross-task attention** (InvPT ECCV 2022, TaskExpert ICCV 2023) runs the backbone
once and lets per-task decoders read each other's features. (ii) **MTMamba++'s cross-task Mamba
(CTM) block** does the same exchange in linear time. Both replace "hope the shared trunk learns
features good for everyone" with *explicit, cheap, learned cross-task communication*. Your own
doc-210 idea (separate cls-token sources: activity from block 14, pose from block 11) is the
zero-parameter version of this and is well-motivated by this literature.

**A5 - Detection necks/heads in MTL.** The decisive result is **YOLOP (MIR 2022,
DOI 10.1007/s11633-022-1339-y)**: on BDD100K 3-task joint training a **grid/anchor-free one-stage
detector loses only -0.4 AP** under MTL (76.9->76.5) while a **two-stage Faster R-CNN collapses
-4.7 AP** (67.3->62.6) and drags segmentation down with it. Your ASD head is YOLOv8-style (FPN
P3/P4/P5, decoupled cls+reg, DFL, TAL) - exactly the family YOLOP shows shares near-losslessly.
**So your ~0.20 mAP is not a "detection-in-MTL law"; it is a fine-grained-classification ceiling +
sparse-label + starvation problem** - consistent with doc-212. Full treatment in Section C.

**A6 - Multimodal-as-MTL.** Out of scope for surfaced literature (single-RGB setting); closest
analogue is EgoVLP egocentric video-language pretraining (Section H), which is sequential transfer.

**A7 - The MTL>ST "holy grail."** Not achieved anywhere for a heterogeneous 4-task video set.
FairGrad beats single-task on all 3 NYUv2 dense tasks (strongest published "beats ST on all
tasks"), but they are commensurate dense predictions. On heterogeneous/high-count benchmarks
(CelebA-40, QM9-11) even the best methods post *positive* (worse-than-ST) dm%. **The holy grail is
unclaimed in your regime - the opportunity, not a reason for pessimism.**

**A8 - Shared / task-specific parameter ratio.** **No benchmarked optimal ratio exists.** Evidence
cuts both ways: Polyhistor/VMT-Adapter show <10% (even <1%) task-specific params suffice for
positive transfer on *cooperative* tasks (arguing your ~23% is generous); MoE argues the opposite
for *competing* tasks - more disjoint capacity converts negative->positive transfer. **Synthesis:
the right task-specific fraction scales with task conflict, not a universal constant.** For your
severe conflict, *increase* effective task-specific capacity for starved heads (low-rank experts)
rather than shrink it. Report "no universal optimum; conflict-dependent."

**Task-grouping - the premise test (A7/A8 corollary).** Before defending "one model for four
tasks," test that premise. **Standley et al. (ICML 2020, arXiv:1905.07553)** shows grouping can
beat both one big MTL net and many ST nets on the accuracy/latency trade-off; **TAG (Fifty, NeurIPS
2021)** measures *inter-task affinity* (how one task's gradient step changes another's loss) in a
single joint run - a ready-made diagnostic for whether activity/PSR are *actively harmed* by
detection/pose vs merely under-weighted. **Actionable:** run TAG-style affinity on your current
joint model. If detection-vs-activity or pose-vs-PSR are strongly negative, a **2-group split**
(geometric {pose, detection} vs temporal {activity, PSR}) *still satisfies the ~2x efficiency
claim* and may be the more defensible AAIML architecture than forcing all four into one trunk.

---

## Section B - Optimization & Loss Weighting (answers B1-B7)

**How to read the numbers.** Field-standard metric **dm%** = average per-task % change vs
single-task, **lower is better, negative beats ST**. Two cautions: (i) dm% is an *average* and can
hide a regressed task (CAGrad reaches NYUv2 dm ~ +0.20 while surface-normal error *worsens*); (ii)
magnitudes cluster tightly - the whole optimizer frontier spans ~+5% (naive scalarization) to ~-5%
(best 2024 method) on NYUv2, i.e. best case ~5% *average* gain, consistent with doc 213.

**Canonical head-to-head (Cityscapes 2-task and NYUv2 3-task, dm% lower=better):**

| Method | Cityscapes dm% | NYUv2 dm% | backward passes | in doc 213 |
|---|---|---|---|---|
| Linear scalarization (LS) | 22.60 | +5.59 | 1 | - |
| Kendall UW | 5.89 | - | 1 | yes |
| RLW (random) | 24.38 | - | 1 | new |
| DWA | 21.45 | - | 1 | yes |
| MGDA | 44.14 | - | k | yes |
| PCGrad | 18.29 | - | k | yes |
| GradDrop | 23.73 | - | k | yes |
| CAGrad | 11.64 | +0.20 | k | yes |
| IMTL-G | 11.10 | - | k | yes |
| MoCo | 9.90 | +0.16 | k | new |
| Nash-MTL | 6.82 | -4.04 | k | yes |
| **FAMO** | 8.13 | **-4.10** | **1** | yes |
| **FairGrad** | **5.18** | **-4.96** | k | new |

The 2022-2024 frontier (Nash-MTL -> FAMO -> FairGrad) converges around **-4 to -5% dm% on NYUv2**.

**B1 - Alternatives to Kendall + caps.** Most important is **FAMO (NeurIPS 2023)**: balances tasks
so losses decrease at approximately equal rates using only the *aggregated* gradient plus a
log-space weight update - **O(1) time/memory, single backward pass, no per-task gradients** -
matching Nash-MTL-class NYUv2 results (-4.10). A drop-in replacement for or complement to your
Kendall+EMA weighting. Cheaper magnitude-balancers: **IMTL-L** (closed-form per-task scalar, single
pass, a cousin of Kendall); **MetaBalance (WWW 2022)** rescales auxiliary-task gradient magnitudes
to match the target per parameter block - **directly targets your 312x activity-vs-PSR domination**;
**DB-MTL (2023)** balances loss scale and gradient magnitude at near-FairGrad level.

**B2 - Gradient surgery beyond PCGrad.** Strongest *reported*: **FairGrad (ICML 2024,
arXiv:2402.15638)** - alpha-fair resource allocation (generalizing Nash-MTL); **best published dm%
on both Cityscapes (5.18) and NYUv2 (-4.96)**, and its "stop the dominating task starving the
struggling task" framing is *the closest published analogue to Kendall-collapse.* **Aligned-MTL
(CVPR 2023, arXiv:2305.19000)** uses the gradient system's condition number + SVD alignment with
**provable convergence to a pre-specified task-weight optimum** (you keep trade-off control).
**SDMGrad (NeurIPS 2023)** steers toward a target direction while conflict-averse. **All three cost
k backward passes (k=4 -> 7-day run becomes weeks).** TIME-friendly exception: **Recon (ICLR
2023)** profiles which layers produce the most conflict (one offline pass), makes those layers
task-specific, then trains normally - **no per-step overhead**; "split the most conflicted MViT
blocks into task branches" is directly actionable.

**B3 - MTL optimizers / MGDA / FAMO.** MGDA is weak (Cityscapes dm 44.14) and expensive. FAMO is
the standout (B1) - a cheap gradient-balancing method: yes, FAMO.

**B4 - Curriculum / staged training.** Literature lacks standardized dm% comparisons for staged
schedules, so no quantitative MTL/ST claim is reportable. Cheap defensible version: warm up starved
heads (activity, PSR) with the backbone down-weighted, then release - a schedule on your existing
capped-Kendall weights, zero extra backward passes.

**B5 - Methods that beat ST on all tasks.** Only FairGrad on NYUv2's 3 commensurate dense tasks
(seg 38.80 vs 38.30, depth 0.557 vs 0.675, normals 24.55 vs 25.01 - seg margin within noise). **No
published method establishes MTL/ST >= 1.0 across 4 heterogeneous heads on video.**

**B6 - Meta-learning the weighting.** **Auto-Lambda (TMLR 2022)** meta-learns task weights by
bilevel optimization, encoding asymmetric task-help (~2x per-step cost). **MetaBalance (WWW 2022)**
is the cheaper, more targeted choice for your primary-plus-auxiliaries framing (pose is headline).

**B7 - 2025-2026 SOTA.** **ConICGrad / UPGrad (arXiv:2502.00217)**, **PAMM (arXiv:2511.14503)**,
**DB-MTL** are the current frontier on the same suite, competitive with FairGrad but
**unreplicated** - cite for currency, do not build on.

**B - the counter-evidence you must defend against (critical).** Three studies argue the
fancy-optimizer premise is weaker than it looks: **Kurin et al. (NeurIPS 2022, arXiv:2201.04122)**
- plain sum-of-losses + standard regularization matches/beats complex optimizers while avoiding
per-task-gradient overhead (MTL optimizers act as implicit regularization); **Xin et al. (2022)** -
MTL optimizers don't improve beyond a well-tuned linear scalarization (corroborated by Hu et al.
NeurIPS 2023); **Elich et al. (GCPR 2024, arXiv:2311.04698)** - the "gradient conflict causes MTL
failure" narrative is more nuanced; conflict magnitude does not cleanly predict final performance.
**Since your contribution is a gradient-pathology diagnosis, these reviewers are your biggest
threat.** Pre-empt them: benchmark capped-Kendall against a *properly regularized* scalarization
baseline **and** FAMO at matched wall-clock. If capped-Kendall matches FAMO and both beat
regularized scalarization *on the starved heads specifically*, that is a defensible, reviewer-proof
result needing no k-backward-pass optimizer.

**Applicability ranking for IndustReal (binding constraint = TIME):**
1. **FAMO** - single backward pass, O(1), best benefit/cost. Try first.
2. **RLW as a control; MetaBalance / IMTL-L** for magnitude balancing (MetaBalance targets 312x gap).
3. **Recon** - offline conflict-profiling -> split most-conflicted MViT layers; no per-step cost.
4. **FairGrad / Nash-MTL / Aligned-MTL** - strongest reported, closest framing, but k passes; only
   if amortized (update weighting every N steps) or restricted to the last shared block.
5. **Auto-Lambda / RotoGrad** - ~2x per step or extra params; RotoGrad's per-task feature rotation
   is one of the few ideas designed for *heterogeneous* tasks.

---

## Section C - Detection inside MTL (answers C1-C6)

**C1 - Why detection degrades most.** Detection loss couples localization (regression) with
fine-grained classification and is dominated by a small set of hard, sparsely-labelled foreground
locations (~18% of your frames carry a box, ~1 box/frame). A dense per-frame co-task (pose) supplies
a large, smooth, constantly-flowing gradient that shapes the backbone toward its own geometry; the
sparse detection signal competes intermittently. Critically, though, in your measured backbone
gradient norms (PSR 3.18 >> detection 0.48 ~ pose 0.44 >> activity 0.010) detection is *not*
gradient-starved - it sits above activity and on par with the healthy pose head. So detection's low
mAP is driven by fine-grained-classification difficulty + label sparsity, **not** by the
magnitude-starvation mechanism that crippled activity; the two heads underperform for different
reasons, and the detection fix is resolution/assignment/data (Section C3, C6), not gradient
rebalancing.

**C2 / A5 - Detection heads that survive a shared backbone.** **YOLOP (MIR 2022,
DOI 10.1007/s11633-022-1339-y)** is the key evidence: grid/anchor-free one-stage detector -0.4 AP
under 3-task MTL vs two-stage Faster R-CNN -4.7 AP. Your YOLOv8-style head is the lossless family.
The DETR/RT-DETR family (query-based, NMS-free, single forward) is the other MTL-friendly option,
but **no controlled MTL/ST retention number for RT-DETR inside a multi-task video trunk was found**
(not reported) - the transferable evidence remains YOLOP's grid-vs-region contrast.

**C3 - Small objects (~20px at 224px on a 14x14 grid).** No surfaced paper isolates small-object
retention under MTL specifically. The generic levers (higher-resolution feature maps, P2 inclusion,
detection-specific FPN branch) are architecture edits, not MTL findings; the YOLOP result implies
the head family is fine, so pursue resolution/assignment (Section C6) rather than a head swap.

**C4 - Cross-task feature exchange for detection.** Same mechanisms as A4 (decoder cross-attention /
cross-task Mamba). Detection-conditioned heads (RoI-Align on detection features to condition other
heads, per your doc-210) is the zero-to-low-param version.

**C5 - mAP-retention >85% cases.** YOLOP is the cleanest (retention ~99.5% for the one-stage
detector). No video/egocentric MTL detection-retention number exists to cite.

**C6 - Anchor assignment (TAL / ATSS / SimOTA).** Your TAL (cls^1.0 * iou^6.0, top-k=10) is a
reasonable choice; the literature does not show an MTL-specific assigner advantage. Bottleneck is
class discrimination, not assignment - do not spend the budget here.

## Section D - Activity recognition in MTL (answers D1-D6)

**D1 - Long-tail 75-class recovery.** The standard toolkit, with the critical caveat that **almost
every method is single-task-validated and MTL-retention is unmeasured** - which is itself a finding.
Prefer **head-only** methods because they never depend on the shared backbone cooperating:
- **Balanced Softmax (NeurIPS 2020, arXiv:2007.10740)** - analytic train/test label-shift correction
  inside the softmax (additive log-prior). A **drop-in loss change on the activity head only**,
  preserves single forward pass. (Note: conflicts with label smoothing - drop your 0.05 smoothing if
  you adopt it.)
- **Decoupling: cRT / LWS / tau-norm (ICLR 2020, arXiv:1910.09217)** - train the encoder with
  instance-balanced sampling, then retrain/rescale *only the classifier* with class balancing. The
  classifier-retraining half is a cheap, post-hoc, head-only rebalance ideal for a shared trunk.
- **LDAM-DRW (NeurIPS 2019, arXiv:1906.07413)** - label-distribution-aware margins + deferred
  reweighting; the DRW schedule (train normally, reweight late) fits your EMA-normalized loop.
- **"Use Your Head" (CVPR 2023, arXiv:2304.01143)** - long-tail *video* recognition on
  EPIC-KITCHENS-100, the closest regime match; documents that image long-tail fixes underperform on
  video because clip sampling interacts with class frequency.

**D2 - Temporal pooling (TSM/SlowFast) vs CLS-token pooling.** The action-recognition literature
(TSM ICCV 2019, SlowFast ICCV 2019) consistently uses *temporal pooling over per-frame features*
rather than a single global token. A pooled/attention head gives a **denser, higher-variance
gradient path** into the backbone than one CLS token - a plausible partial remedy for your 312x
gradient-norm gap, since a single token bottlenecks the gradient the head can push. **No controlled
"CLS-token vs temporal-pool -> backbone gradient magnitude in MTL" ablation exists** (not reported);
this is a hypothesis to test, matching your doc-210 activity-head-redesign idea (Temporal Attention
Pool: 49 spatial gradient sources vs 1).

**D3 - Verb-noun factorization.** EPIC-KITCHENS-100 (below) factorizes action into verb+noun as a
shared-encoder multi-head setup - a template if you ever decompose your 75 classes.

**D4 - 95%-retention conditions.** Published activity-MTL retention is 90-95% *when data is
sufficient* (OmniSource-style). Your collapse to ~0% is not normal MTL interference - it was a
gradient-path bug (doc-209 in-place tensor assignments), now a recovery problem, not an
interference problem.

**D5 - Cross-task sharing for activity.** Detection-conditioned or PSR-conditioned activity features
(shared temporal context) is the zero-param idea; no published MTL number.

**D6 - MTL-compatible augmentation.** Adopt **class-mean (balanced) recall** as the activity metric
(EPIC-KITCHENS-100, IJCV 2022, DOI 10.1007/s11263-021-01531-2) so a majority-class collapse is
*visible* in the headline number - your single most important measurement change for activity.

## Section E - Procedural State Recognition in MTL (answers E1-E6)

**This is genuine white space: no published work measures procedural-state / event-F1 with a
backbone shared across other tasks.** Your MTL-PSR result would be novel. The transferable evidence
is architectural:

- **E2 (per-frame -> event-F1 bridge) - MS-TCN (CVPR 2019, arXiv:1903.01945).** The key result for
  your metric mismatch: on 50Salads a single-stage TCN reaches 78.2 frame-accuracy but only
  **F1@10 = 27.0**, while multi-stage refinement + smoothing loss reaches **F1@10 = 76.3 at 80.7
  frame-accuracy** - frame-accuracy barely moves (+2.5) while segmental/event F1 nearly **triples**.
  The gain is entirely from suppressing over-segmentation (spurious short flips), which is exactly
  what event-F1@+-3-frame punishes and exactly the failure mode of a per-frame classifier at <1%
  transition rate. **Stack a multi-stage temporal-refinement head (or the smoothing loss alone) on
  your causal transformer.**
- **E5 (temporal transformers/TCN) - ASFormer (BMVC 2021, arXiv:2110.08568)** extends MS-TCN with
  attention; a natural upgrade path from your current causal transformer head.
- **E3 (monotonicity as inductive bias).** Your 0->1-never-1->0 constraint maps onto ordinal /
  cumulative-link heads (ordinal regression as K-1 monotone binary decisions) or an explicitly
  monotone activation. This **encodes the accumulation prior structurally** so the head cannot emit
  a physically impossible 1->0 transition and needs no data to learn it. Not applied to
  procedural-state-in-MTL specifically (not reported), but a standard construction worth adding.
- **E6 (window-to-recording aggregation from T=8).** No published MTL-specific method; the
  refinement head operating over stacked windows is the practical route.

**Bottom line E:** no literature target for "PSR retention in MTL" exists - you are first. The
evidence-backed levers are (1) multi-stage temporal refinement + smoothing loss (MS-TCN, quantified)
and (2) a monotone/ordinal head. Both preserve the single forward pass.

## Section F - Efficiency & Reporting Methodology (answers F1-F5)

Grounded in **Vandenhende et al., MTL for Dense Prediction: A Survey (IEEE TPAMI 2021,
DOI 10.1109/TPAMI.2021.3054719)** and the conventions later dense-MTL papers follow:

1. **Report dm% as the headline MTL metric**, with the sign convention explicit (your pose head is
   lower-is-better in degrees; detection/AR/PSR are higher-is-better). This turns "beats ST on N of
   4 tasks" into one auditable number. Your composite (geometric mean of per-head MTL/ST ratios, doc
   221) is compatible and complementary.
2. **Matched-backbone comparison is mandatory** (hold backbone fixed, vary head/optimization). This
   is your Ablation A (ST baselines on the same MViTv2-S) - the single most important missing
   experiment per doc 212. Vandenhende notes <30% of MTL papers report a matched ST baseline;
   reporting it is a credibility differentiator.
3. **FLOPs reported once for the shared encoder + per-task head overhead** - do NOT multiply encoder
   FLOPs by task count. This is the single most important reporting choice for your "single forward
   pass / ~4x latency" claim: state clearly the shared encoder is evaluated once, then give the
   4x-ST baseline separately for the efficiency delta.
4. **Params: total / shared-trunk / per-head split** as standard columns. Your ~34.5M shared + small
   heads (~42.7M total) vs 4x single-task is a strong efficiency story.
5. **params-vs-dm% Pareto plot** with the 4-ST cluster - the visual your hypothesis needs, showing a
   single MTL point dominating the ST cluster on the params axis at comparable performance.
6. **Negative-result honesty is now expected** (Xin/Kurin): report a strong tuned-scalarization
   baseline and frame capped-log-var as addressing a *specific diagnosed pathology*, not a general
   optimizer win.

## Section H - Foundation Models, SSMs, Wildcards (answers H1-H6)

> Numeric values in this section are the Track-4 author's domain-knowledge recollection, **pending
> primary-source verification** - check each figure against the cited paper before use.

**H3 - Video/image foundation models as MTL backbones.** No published video foundation model reports
a shared-encoder result on 4 heterogeneous heads of the IndustReal type. All multi-task evidence is
**sequential transfer** (pretrain once -> fine-tune each task separately) or **frozen-feature
probing**, never simultaneous joint MTL from a shared trunk. VideoMAE-V2 / UMT / InternVideo2 raise
K400 top-1 by ~5-11 points over MViTv2-S (81.0%), which lifts every head's *ceiling* - but nothing
shows a stronger trunk reduces gradient competition (ceiling != gap). Only small/base variants
(InternVideo2-S/B, Hiera-B ~51M, VideoMamba-M ~74M) are budget-compatible; the giant variants
(VideoMAE-V2-g ~1B, InternVideo2-6B) break the efficiency claim. **EgoVLP/EgoVLPv2** (Ego4D
egocentric pretraining) is the best domain match to your HoloLens-2 setting but still evaluates
tasks separately. **The frozen-trunk foil: DINOv2** supports many dense tasks from a *frozen*
encoder with lightweight heads - the strongest "one encoder -> many tasks" evidence, but it works by
*removing* inter-task gradient competition by construction. If your thesis is "a *shared, co-adapted*
trunk boosts all tasks," DINOv2 is a *foil* you must address head-on, not support.

**Recommendation:** keep MViTv2-S as the headline backbone (WACV comparability); report a *single*
backbone-swap ablation (VideoMamba-M or InternVideo2-B) showing the ceiling moves but the gap
persists - which *strengthens* the "Kendall-collapse is an optimization problem, not a capacity
problem" argument.

**H4 - SSMs (Mamba/RWKV/Hyena) as backbones.** **VideoMamba (ECCV 2024, arXiv:2403.06977)**:
bidirectional Mamba video encoder, linear complexity, ~85% K400 (M, distilled), ~6x lower GPU memory
on long videos, ~74M params. **But no joint-MTL result exists** - SSMs are a **latency/memory lever,
not an MTL lever.** Attractive for your time-bound RTX setup, but treat as an efficiency ablation,
not a Kendall-collapse fix. (The MTMamba++ result in Section B is an SSM *decoder*, which is the more
promising SSM direction for you.)

**H1/H2/H5/H6 - remaining wildcards.** No surfaced evidence for multi-agent-RL or NAS closing the
MTL->ST gap in this regime. The ideal-2026-system synthesis is Section 4 below.

---

## Section 4 - Prioritized Recommendation Shortlist (mapped to the four heads + 30-day roadmap)

**Framing decision first (highest leverage, ~1 GPU-day).** Before committing to "one trunk for four
tasks," run a **TAG-style inter-task affinity** measurement (Fifty NeurIPS 2021) on one joint run.
It tells you whether activity/PSR are *actively harmed* by detection/pose or merely under-weighted.
If two clusters are strongly antagonistic, a **2-group split** ({pose, detection} geometric vs
{activity, PSR} temporal) still delivers the ~2x efficiency claim and is more defensible than
forcing all four into one trunk. This directly de-risks the paper's central premise.

**Per-head shortlist (evidence-backed, single-forward-pass-preserving, ordered by benefit/cost):**

| Head | Problem | Recommendation (method, cite) | Cost | Expected effect |
|---|---|---|---|---|
| **Global / weighting** | 312x activity-vs-PSR gradient gap; Kendall-collapse | **FAMO** (NeurIPS 2023) replace/augment Kendall+EMA; **MetaBalance** (WWW 2022) to rescale starved-head grads | 1 backward pass (FAMO); low | Balances decrease rates without k passes; MetaBalance targets the exact 312x gap |
| **Global / geometry** | conflicted shared layers | **Recon** (ICLR 2023): offline-profile most-conflicted MViT blocks -> make task-specific | 1 offline pass, no per-step cost | Removes worst conflict cheaply under TIME budget |
| **Activity** | long-tail collapse | **Balanced Softmax** (NeurIPS 2020) head-only loss swap (drop label-smoothing); **cRT** classifier-retrain (ICLR 2020) | trivial | Head-only rebalance independent of shared trunk |
| **Activity** | 1-token gradient bottleneck | **Temporal Attention Pool** head (TSM/SlowFast-style, doc-210 idea) | +~0.7M params | 49 spatial grad sources vs 1; denser backbone gradient |
| **Activity** | metric hides collapse | Report **class-mean recall** (EPIC-KITCHENS-100, IJCV 2022) | free | Majority-collapse becomes visible in headline |
| **PSR** | per-frame -> event-F1 over-segmentation | **MS-TCN multi-stage refinement + smoothing loss** (CVPR 2019) on top of causal transformer | +small | F1@10 27->76 template at constant frame-acc (white space in MTL) |
| **PSR** | impossible 1->0 flips | **monotone / ordinal head** (cumulative-link) | ~0 params | Encodes accumulation prior structurally |
| **Detection** | fine-grained ceiling, NOT starvation | resolution/assignment (P2 inclusion, higher-res feature map); head family already lossless (YOLOP) | arch edit | Addresses the real bottleneck; do not spend budget on assigner/optimizer |
| **Pose** | healthy (MTL/ST ~0.77) | move Gram-Schmidt into model -> pure geodesic; no MTL change | ~0 | Marginal; protect, don't perturb |
| **Backbone (ablation only)** | ceiling vs gap | one swap to **VideoMamba-M** or **InternVideo2-B** | 1 run | Show ceiling moves, gap persists -> strengthens optimization-not-capacity thesis |

**Reviewer-proofing (mandatory, from B counter-evidence + F).** Benchmark capped-Kendall against
(a) a **properly-regularized tuned scalarization** baseline and (b) **FAMO**, at matched wall-clock,
reporting per-head deltas - especially on the starved heads. This pre-empts Kurin/Xin/Elich, who
argue tuned scalarization matches fancy optimizers. Report **dm%** + matched-backbone ST baselines
(your Ablation A - the single most important missing experiment) + FLOPs-counted-once + a
params-vs-dm% Pareto plot.

**Mapping onto your 30-day roadmap (doc 226 gates):**

- **Phase 1 (-> G1 day 3):** TAG affinity run (framing decision); swap activity metric to class-mean
  recall; Balanced Softmax on activity head. All cheap, all de-risk the two weakest heads.
- **Phase 2 (-> G2 day 7):** FAMO as the weighting method (single backward pass - fits TIME budget);
  MetaBalance as fallback for the 312x gap; Temporal Attention Pool activity head.
- **Phase 3 (-> G3 day 20):** MS-TCN refinement + smoothing head on PSR; Recon offline conflict-
  profiling -> split most-conflicted MViT blocks; matched-backbone ST baselines (Ablation A).
- **Phase 4 (-> G4 day 26):** one backbone-swap ablation (ceiling-vs-gap); tuned-scalarization +
  FAMO reviewer-proofing table; params-vs-dm% Pareto figure. Reserve FairGrad/Nash/Aligned-MTL
  (k backward passes) ONLY if a GPU frees up and amortized weighting is acceptable.

**One-sentence honest verdict.** No published method reaches MTL/ST >= 1.0 across four heterogeneous
video heads - the holy grail is unclaimed in your regime, so the defensible AAIML contribution is
(1) the Kendall-collapse *diagnosis* validated against FAMO and regularized-scalarization baselines,
(2) *comparable-or-better on the healthy/recoverable heads* at ~2x parameter efficiency and
single-pass latency, and (3) the first procedural-state (PSR) and egocentric-pose results inside a
shared-backbone MTL model - not a universal MTL>ST claim.

---

## Appendix - Evidence table

The full 52-method evidence table (43 not previously in doc 213) is saved alongside this report as
`mtl_methods_evidence_table.csv` with columns: method, venue_year, arxiv_doi, family, benchmark,
task_count, reported_MTL_ST_or_dm, single_forward_pass, per_task_backward, applicability_industreal,
in_doc213.
