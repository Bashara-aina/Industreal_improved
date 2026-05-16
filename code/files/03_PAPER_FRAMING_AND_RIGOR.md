# 03 — Paper Framing, Evaluation Rigor, and What to Claim

Doc 01 closed the per-target gaps. Doc 02 made them stay closed. This document is about turning the resulting numbers into a paper that survives review.

The current `popw_paper.tex` describes an architecture that is implemented and correct (post-bug-fix). The current `popwbenchmark.tex` lists targets that the Doc 01 plan can clear. The remaining work is to (a) make sure the comparisons are protocol-correct, (b) handle the IKEA-ASM gap honestly, and (c) frame the contribution where POPW is actually strong, not where the user wishes it were strong.

---

## A. Protocol correctness — the trap that sinks multi-task papers

Reviewers do not care that you got 0.92 PSR F1. They care whether your 0.92 is comparable to STORM-PSR's 0.901. The only way they trust your number is if every protocol detail matches.

The IndustReal targets in your `popwbenchmark.tex` come from three different papers with three different evaluation conventions. Mixing them is the most common failure mode.

### A.1 — Detection mAP@0.5 (vs YOLOv8m, 83.80%)

The IndustReal paper reports YOLOv8m at "83.80% mAP@0.5". Buried in their Section 5.2:
- Evaluated on **frames containing ground-truth bounding boxes only**, not entire videos
- Two reported numbers in their Table 3: `mAP (b-boxed) = 0.838` and `mAP (entire videos) = 0.641`
- The 83.80% is the b-boxed number — "mAP", not "mAP@0.5"

**Implication:** what they call "mAP" is the COCO-style average over IoU thresholds 0.5:0.95:0.05 *on annotated frames only*. This is materially different from "mAP@0.5 across all test frames". If you report mAP@0.5 (all frames), your number will look lower than 83.80% even if your model is better.

**What to do:**
1. Report **two** numbers in your headline table:
   - `mAP (b-boxed)` — the IndustReal paper's protocol, comparable to their 83.80%
   - `mAP@0.5 (all frames)` — your own number, for reference
2. Also report `mAP@[0.5:0.95]` as the COCO-style fair comparison
3. State the protocol in the table caption: "Reported on frames with ground-truth bounding boxes, following Schoonbeek et al. (2024) Table 3"

If you report just "mAP@0.5 (all frames)" and call it 0.85, a careful reviewer will spot that you're not actually beating 0.838 on the same protocol.

### A.2 — Activity Top-1/Top-5 (vs MViTv2, 66.45% / 88.43%)

The IndustReal paper's MViTv2 is **multi-modal** — RGB + visible-light + stereo (their Table 2, last row). Their RGB-only MViTv2 is 65.25% Top-1 / 87.93% Top-5.

**Implication:** if you only have RGB, the "fair" comparison is against the 65.25% / 87.93% RGB-only numbers, not the 66.45% / 88.43% multi-modal numbers. If you compare RGB-only POPW against multi-modal MViTv2 and you lose by 0.5%, that's not a defeat — it's an apples-to-oranges comparison where you're using less information.

**What to do:**
1. State explicitly in the activity table caption: "POPW uses RGB only. MViTv2 multi-modal uses RGB + VL + stereo (cited number 66.45%); MViTv2 RGB-only is 65.25% (cited but in same table)."
2. Report POPW vs **both** MViTv2 numbers
3. If POPW > 66.45%, state "POPW (RGB) outperforms MViTv2 (RGB+VL+stereo)" — that's a strong claim.
4. If POPW > 65.25% but not > 66.45%, state "POPW (RGB) outperforms MViTv2 (RGB)" — still a valid claim.
5. Never compare POPW (RGB) only against the 66.45% number without flagging the modality gap.

### A.3 — PSR F1/POS (vs STORM-PSR, 0.901 / 0.812)

STORM-PSR (CVIU 2025) uses **±3-frame tolerance**. The IndustReal paper's B3 baseline (0.883 F1) uses **±5-frame tolerance**. These are different metrics measured under different settings.

**Implication:** a model can clear B3's 0.883 at ±5 and still fall short of STORM-PSR's 0.901 at ±3. If you report your number at ±5 and compare against STORM-PSR's ±3 number, you've inflated your perceived performance.

**What to do:**
1. Report PSR F1 and POS at **both** tolerances (±3 and ±5) in every table. Your `evaluate.py` already supports the `tolerance_frames` argument.
2. Compare:
   - Your ±5 number vs B3's 0.883 → "beats B3 baseline"
   - Your ±3 number vs STORM-PSR's 0.901 → "beats STORM-PSR" (only if the ±3 number actually clears 0.901)
3. Caption: "We report at ±3 (STORM-PSR convention) and ±5 (IndustReal paper convention)."

If you can't clear STORM-PSR's 0.901 at ±3 even with sequence-mode training, the honest claim is "We beat the IndustReal paper's strongest published baseline (B3, F1=0.883 at ±5) by X points; we are competitive with STORM-PSR (F1=0.901 at ±3) within Y points."

That is still a publishable claim. It is **not** a state-of-the-art claim, and pretending it is will get the paper rejected.

### A.4 — Activity protocol: clip-level vs frame-level

The IndustReal paper evaluates Activity on **action clips** (Section 5.1). Each clip is one annotated action, the model predicts a single class for the entire clip. Their MViTv2 takes 16 frames sampled from the clip and outputs one class.

**Implication:** if your `evaluate.py` reports per-frame Top-1, you're solving a harder problem than they are. Per-frame Top-1 will be lower than clip-level Top-1 (typically by 1.5–3 points) because most frames near action boundaries are ambiguous.

**What to do:**
1. The headline `act_top1` you report **must be clip-level** to be comparable to 66.45%.
2. Your `evaluate.py` has `_compute_clip_level_accuracy()` — confirm it's wired into the headline metric.
3. You can additionally report frame-level Top-1 as a secondary metric, but the comparison number is clip-level.

### A.5 — Combined protocol checklist for the paper

Add an "Evaluation Protocol" subsection to your paper (it's missing in `popw_paper.tex`). State explicitly:

1. **Detection**: `mAP` reported on annotated frames only (matches IndustReal Table 3); we additionally report `mAP@0.5` and `mAP@[0.5:0.95]` for completeness.
2. **Activity**: clip-level Top-1 and Top-5 with 16-frame uniform sampling per clip. RGB modality only.
3. **PSR**: F1 and POS at ±3-frame and ±5-frame tolerances; we report both.
4. **Head pose**: angular MAE in degrees (vectors L2-normalized before dot product) for forward and up; position MAE in mm.
5. **Assembly state F1@1**: top-confidence detected class vs ground-truth state, per frame.
6. **Error verification AP**: AP score for binary (error/no-error) classification, where score = 1 − confidence(expected state).

Without this section, half your numbers are not comparable to your competitors.

---

## B. The IKEA-ASM problem

The current papers (`popwbenchmark.tex` and `popw_paper.tex`) describe and target IKEA-ASM benchmarks. **There is no IKEA-ASM dataset module in your codebase.** You only have `industreal_dataset.py`. This is a real gap between the paper's claims and the implementation.

You have three options. The choice shapes the paper's positioning.

### B.1 — Option A: drop IKEA-ASM from the paper (safest)

Reframe the paper as IndustReal-only. Title becomes:

> "POPW: A Unified Multi-Task Architecture for Egocentric Industrial Assembly Recognition"

Or:

> "POPW: Pose-Conditioned Multi-Task Learning on IndustReal"

**Pros:** the paper's claims match the implementation. No deferred work.
**Cons:** you lose half the benchmark coverage; the architecture's "dual-dataset adaptability" angle disappears.

### B.2 — Option B: implement IKEA-ASM dataset module before submission

Build an `ikea_asm_dataset.py` mirroring `industreal_dataset.py` but for the IKEA-ASM data layout. The architecture is already configured to handle the dimensional differences (`TRAIN_HEAD_POSE=True` for IKEA-ASM body pose, `NUM_CLASSES_DET=7`, `NUM_CLASSES_ACT=33`, `IMG_SIZE=(640, 480)`). The dataset module is the missing piece.

**Estimated effort:** 3–5 days for a usable module, plus 4–6 days for IKEA-ASM training runs, plus 2 days of multi-seed.

**Pros:** the paper claims hold. Strong positioning ("works on two distinct datasets with the same architecture").
**Cons:** ~2 weeks of engineering work and compute.

### B.3 — Option C: paper-claim IKEA-ASM, mark as "future work" or "in progress"

Keep IKEA-ASM in the related-work and architecture-description sections (showing the architecture *can* handle IKEA-ASM), but in the experiments section state that IKEA-ASM evaluation is in progress and will be in the camera-ready or supplementary.

**Pros:** preserves the architectural narrative without the engineering cost.
**Cons:** reviewers may push back on this. "Future work" in an experiments section is usually a signal of incompleteness. The paper is then judged on IndustReal alone, which is the same as Option A but with extra writing effort.

**Recommendation:** Option A unless you have 2 weeks of compute and engineering bandwidth. Option B is the paper-strongest path. Option C is a hedge that often costs more than it saves.

---

## C. What to claim and what to caveat

Honest framing of POPW's contribution. Get this right and the paper is strong even if a target slips.

### C.1 — Claim categories

**Strong claims (defendable if Doc 01 plan succeeds):**
- "POPW is a unified multi-task architecture that performs detection, head pose estimation, activity recognition, and PSR in a single forward pass."
- "POPW exceeds the strongest published single-task baselines on every IndustReal headline benchmark (with the protocols stated in Section 4)."
- "PoseFiLM and HeadPoseFiLM provide a novel two-stage feature conditioning mechanism for cross-task information flow."

**Conditional claims (defendable only if specific conditions hold):**
- "POPW outperforms STORM-PSR on PSR" — only if your ±3-frame F1 actually clears 0.901. Otherwise drop to "competitive with".
- "POPW outperforms MViTv2 on Activity Top-1" — only if your RGB-only number clears the multi-modal 66.45%. Otherwise compare against RGB-only 65.25%.
- "POPW achieves real-time streaming inference on commodity hardware" — only if your streaming FPS measurement (already in `efficiency_report.py`) clears ~25 FPS at 720p.

**Claims to avoid:**
- "POPW is more efficient than [single-task model]" — at 50–60M params and ~75 GFLOPs at 720p, POPW is heavier than YOLOv8m alone. The efficiency narrative is "one model for five tasks", not "fewer parameters than a single-task baseline".
- "POPW is a SOTA model" — without IKEA-ASM and with PSR's per-frame training caveat, this is a stretch.
- Anything implying you re-trained the baselines yourself. You're comparing against published numbers; state that explicitly.

### C.2 — The contribution paragraph

Your introduction's contribution paragraph should read something like (adjust to your style):

> "We make the following contributions: (1) POPW, the first unified multi-task architecture for egocentric industrial assembly understanding, performing assembly state detection, 9-DoF head pose estimation, 74-class activity recognition, and 11-component procedure step recognition in a single forward pass on a shared ConvNeXt-Tiny + FPN backbone. (2) A two-stage FiLM conditioning mechanism (PoseFiLM and HeadPoseFiLM) that propagates pose and viewpoint information into activity recognition without gradient interference. (3) Empirical results on IndustReal showing that POPW exceeds or matches the strongest published baseline on every headline benchmark (ASD, Activity, PSR, Assembly State Recognition, Error Verification), under each baseline's reported protocol."

This is honest, specific, and defendable. It does not promise IKEA-ASM (Option A path). It does not promise SOTA. It promises "exceeds or matches", which is what the Doc 01 plan delivers.

### C.3 — The "limitations" section

You should have one. It pre-empts reviewer complaints.

Honest limitations (state them explicitly):
1. **PSR temporal modeling at training time.** "Our PSR head uses a causal Transformer, but the current training pipeline samples one frame per recording per step. This means the temporal modeling is exercised at evaluation only. Sequence-batched training is identified as future work." (Or, if you implement Doc 01 §D, drop this.)
2. **Single-dataset evaluation.** If you go with Option A, state: "Evaluation is conducted on IndustReal. While the architecture admits configuration for IKEA-ASM (different output dimensions, body keypoint pose), full evaluation on that dataset is left as future work."
3. **No comparison against contemporaneous multi-task baselines.** State: "Prior multi-task work in egocentric video focuses on different task combinations (e.g., pose+action). We compare against the strongest single-task baselines for each of the five POPW tasks, since no prior multi-task baseline shares POPW's task set."

These three limitations are the ones a strong reviewer will flag anyway. Pre-empting them is a sign of a careful paper.

---

## D. The specific tables that must appear

`popwbenchmark.tex` already has the right structure. Two adjustments based on Sections A and B above:

### D.1 — Modify Table 3 (IndustReal headline) to add protocol columns

Current structure:
```
ASD: YOLOv8m mAP@0.5 = 83.80
```

Change to:
```
ASD: YOLOv8m mAP (b-boxed) = 0.838  [Schoonbeek 2024 Table 3]
     POPW   mAP (b-boxed) = ?       [our number, same protocol]
     POPW   mAP@0.5 (all) = ?       [our number, alternative protocol]
     POPW   mAP@[0.5:0.95] = ?      [COCO standard]
```

Same expansion for Activity (RGB vs multi-modal) and PSR (±3 vs ±5).

### D.2 — Add an explicit "Protocol Notes" table

A small table near the headline tables that maps each metric to its protocol source paper and any non-default parameters (tolerance, IoU threshold, evaluation frame subset). Reviewers love this; it makes the comparison airtight.

Example row:
| Metric | Protocol from | Key parameter |
|---|---|---|
| ASD mAP | Schoonbeek 2024 §5.2 | b-boxed frames only |
| Activity Top-1 | Schoonbeek 2024 §5.1 | clip-level, 16 uniform-sample frames |
| PSR F1 | Schoonbeek 2025 (STORM) | ±3 frame tolerance |
| ... | ... | ... |

### D.3 — Multi-seed mean ± std

Currently `popwbenchmark.tex` says POPW results "will be reported as mean±std across three random seeds". When you fill in the numbers, follow that promise. Do not report best-of-three or first-seed — the multi-seed convention is mean ± std with at least n=3.

If you only have time for n=2 seeds, state n=2 explicitly and note that the small sample size limits conclusions about variance.

### D.4 — Efficiency table

`popwbenchmark.tex` Table 8 has a row for POPW with empty cells. Fill these from your `efficiency_report.py --baseline_compare` output. Specifically:
- Params (M): about 50–60M depending on whether VideoMAE is on
- GFLOPs (1280×720): about 75–90 depending on VideoMAE
- Batched FPS (RTX 3060): about 12–18
- Streaming FPS (RTX 3060): about 22–30

The streaming FPS number is the most paper-relevant. POPW's feature-bank design makes streaming naturally efficient — competitors (especially MViTv2) must reprocess full clips.

The efficiency narrative for the paper:
> "POPW achieves competitive streaming inference (~25 FPS at 720p on commodity hardware) by virtue of its single-pass design. This is the deployment-relevant metric for industrial assistance applications, where each frame must be processed as it arrives. Competing single-task pipelines, even when each task is faster individually, must be run sequentially or in parallel — yielding combined throughput bounded by the slowest stage."

---

## E. Ablation study — the 6-row table that justifies the architecture

`popw_paper.tex` Section 6 (Experiments) should contain an ablation table. The benchmark beating numbers are necessary but not sufficient — reviewers want to know which architectural pieces contribute what. The minimal table is:

| Configuration | ASD mAP | Top-1 | PSR F1 | Notes |
|---|---|---|---|---|
| Baseline (ResNet-50, single-task heads, no FiLM) | ? | ? | ? | sanity reference |
| + ConvNeXt-Tiny backbone | ? | ? | ? | backbone swap |
| + PoseFiLM (1st stage) | ? | ? | ? | adds body keypoint conditioning |
| + HeadPoseFiLM (2nd stage) | ? | ? | ? | adds head pose conditioning |
| + Causal Transformer PSR (vs MLP) | ? | ? | ? | architectural for PSR |
| + Kendall + staged training | ? | ? | ? | training strategy |
| **POPW-Full** | ? | ? | ? | everything |

Each row is a 30-epoch run at seed=42 (faster than the 100-epoch full runs). Total compute: 6 × 1.5 days = ~9 days. **Plan this in advance** because it is the longest single time block in the entire publication-readiness path.

If you don't have 9 days of compute, narrow to 4 rows: Baseline → +ConvNeXt → +FiLM (both stages) → POPW-Full. Less compelling but defensible.

---

## F. The pre-submission checklist

Before submitting `popw_paper.tex` anywhere, verify each of the following:

### F.1 — Correctness
- [ ] All MASTER_BUG_REPORT bugs verified fixed (run the verification grep checks from §504–516 of that report)
- [ ] Smoke test passes (Doc 02 §A.2)
- [ ] Stage 1 has nonzero `log_var_det.grad` and zero contributions from pose/act/psr losses (Doc 02 §B.1)
- [ ] Stage 3 has nonzero gradients on all four `log_var_t` parameters

### F.2 — Numbers
- [ ] All POPW results are mean ± std over n=3 seeds (or n=2 with explicit caveat)
- [ ] All competitor numbers cited with paper, table, and protocol details
- [ ] `mAP (b-boxed)` reported alongside `mAP@0.5` for ASD
- [ ] PSR reported at both ±3 and ±5 tolerances
- [ ] Activity reported clip-level
- [ ] Head pose reported as angular degrees (forward + up) and position mm
- [ ] Efficiency table includes streaming FPS

### F.3 — Framing
- [ ] No SOTA claims that don't match a specific protocol
- [ ] No "more efficient than X" claims unless X is a multi-task pipeline
- [ ] Limitations section present
- [ ] IKEA-ASM either implemented (Option B) or scoped out of the paper (Option A)
- [ ] Multi-task framing is "one model for five tasks", not "smaller model than competitors"

### F.4 — Reproducibility
- [ ] Code release plan (or repository link) in the paper
- [ ] All hyperparameters in the paper or supplementary
- [ ] All three random seeds documented
- [ ] Hardware (RTX 3060) and approximate wall-clock training time stated

### F.5 — Reviewer pre-empt
- [ ] Why ConvNeXt-Tiny over ResNet-50 — answer in §3.1 of the paper, expand if needed
- [ ] Why Kendall over manual loss weights — short paragraph in §3.4
- [ ] Why staged training — short paragraph in §3.4 with citation to similar work
- [ ] Why FiLM over concatenation — short justification in §3.3
- [ ] Per-class breakdown for Activity and ASD in supplementary (not main)

If every box above is ticked, the paper is in submission shape.

---

## G. The realistic timeline from now

Assuming MASTER_BUG_REPORT fixes are merged and verified:

| Phase | Duration | Output |
|---|---|---|
| Doc 01 §A: synthetic pretraining | 1 day | Pretrained detection checkpoint |
| Doc 02 §A.2: smoke test | 30 min | Sanity-checked staged Kendall |
| Doc 02 §A.3: main run, seed=42 | 3–5 days | Single-seed final checkpoint |
| Doc 01 §D.1+D.2: sequence-mode PSR (if pursuing STORM-PSR target) | +1.5 days | Sequence dataloader + alternate-batch training |
| Doc 02 §A.5: full evaluation, all TTA settings | 1 day | Headline tables filled |
| Decision: stop or continue to multi-seed |
| Doc 02 §A.6: multi-seed (2 more seeds) | 6–10 days | Mean ± std numbers |
| Doc 03 §E: ablation study (6 rows × 30 ep) | 9 days | Ablation table |
| Doc 03 §F: paper polish | 3–5 days | Submission-ready |

**Critical-path total: ~3 weeks** to submission-ready, assuming you skip ablation (acceptable for a workshop or "in progress" submission) or run ablation in parallel.

**Best-case minimal path (Option A, no ablation, single-seed):** ~1 week. Defensible at workshops, marginal at top venues.

**Recommended path (Option A, ablation, multi-seed):** ~3 weeks. Submission-ready for major venues.

The single highest-value place to spend an extra day: **sequence-mode PSR training** (Doc 01 §D). Without it, the STORM-PSR comparison is the weak spot in the entire paper.

The single most-skippable item: **5-fold cross-validation** (`cross_validate.py` exists but is not necessary). Multi-seed (3 seeds) replaces it for typical reviewer expectations.
