# 201 — Opus Round 4 Answer: "MTL Helps, Not Hurts" — Stop Building, Start Measuring

**Date:** 2026-07-10
**Inputs read line-by-line:** 195 (overview), 196 (architecture), 197 (results), 198 (per-head), 199 (paths), 200 (prompt). Prior rounds re-read: 181, 186, 192.
**Ground-truth re-verified against the executed code this round:**
`src/models/mvit_mtl_model.py` (524 lines — PSR/Activity/Detection heads, forward routing), `scripts/train_mtl_mvit.py` (activity class-weights, activity eval loop, PSR loss + T=8 downsample, Kendall/PCGrad), `scripts/train_st.py` (ST baseline entry), `analyses/.../efficiency_audit.md` (measured params), git log (what was actually run).
**Authority rule (unchanged from 181/186/192):** where the round-4 documents (195–200) and the code/measurements disagree, the code and measurements win, and I say so explicitly.

---

## 0. READ THIS FIRST — the one-paragraph answer

**Files 195–200 describe a project that has, once again, answered the wrong question — and this time the answer actively destroys the paper's thesis.** The central question is "does MTL help, not hurt?" That question is answered by **one experiment: MTL vs matched single-task baselines.** Those baselines have **not been run** (there is no `runs/st_*` directory and no ST eval JSON anywhere in the repo — verified). Neither has the **overfit-200 eval probe that 192 §6 named the single most important thing to do first** (`scripts/overfit_50img_cls.py`, `mvp_probe3_psr_ab.py`, `mvp_probe4_tal_vs_3x3.py`, `e8_gradient_diagnostic.py` are all committed as *code* with **zero result files** — verified). Instead, the team spent the round doing exactly what 181/186/192 each warned against — **pattern-matching "below SOTA ⇒ add architecture"** — and shipped a **117.7M model whose 70.9M PSR head (60% of the whole model) is 14–23× larger than the PSR specialist it is supposed to make cheaper (`efficiency_audit.md`: PSR specialist ≈ 3–5M).** The team's *own* audit measures four real specialists at **~100M** and the old MTL model at **46–54M** — a genuine, publishable **~2× efficiency win.** run11 at **117.7M is now LARGER than running four separate specialists.** This is the identical failure mode 192 flagged for the foundation backbone ("it inverts the efficiency claim") — achieved this time through a bloated head instead of a bloated backbone. **The efficiency claim is the paper's spine (181 §4, 186, 192 §7-2). run11 snapped it.** Meanwhile the two "signals" the docs celebrate are both illusory as evidence: (1) PSR loss dropping 1.56→0.17 under **focal-BCE (α=0.25, γ=2) on a task whose positives are <1% of frames is exactly what an all-negatives collapse produces** — it is consistent with F1=0, and 197 §5 / 198 §3.3 half-admit this; (2) activity at **0.58% is *below* uniform random (1.33%)**, which is **not "undertrained" — it is a class-weight collapse or a label/eval misalignment** (I verified the eval is standard argmax and that training applies sqrt-tamed inverse-frequency weights up to 11.71× — a cold head under those weights predicts rare classes and scores below random on a common-class test set). **My recommendation: freeze the architecture, stop adding capacity, and run the two experiments that actually answer the question — the overfit-200 eval probe (½ day) and the four ST baselines (already coded, `train_st.py --task`). Then shrink the PSR head from 70.9M to ≤15M to restore the efficiency spine. Only then write the L2+L3+method paper.** run11's training-loss curves are not evidence for or against MTL; they are theatre until an eval metric and a baseline exist. Do not run11→ep30→paper (Path A). Do Path B, gated on the probe, with a mandatory PSR-head diet.

---

## 1. THE TWO FINDINGS THAT REFRAME EVERYTHING

### Finding 1 — You have not run the experiment that answers your own question.

The title of 195 is *"Can We Prove 'MTL Helps, Not Hurts'?"* You cannot prove it — or disprove it — from anything in run11. "MTL helps, not hurts" is **definitionally a comparison against single-task**:

- **helps** ⇒ MTL metric **>** ST metric (positive transfer) on ≥1 head
- **doesn't hurt** ⇒ MTL metric **≈** ST metric (≥~0.9, benign sharing) on the rest
- **more efficient** ⇒ MTL params/latency **<** running the ST set

Every one of those three is a ratio whose denominator is a single-task number **you have not measured.** The entire run11 apparatus — TAL, the 6-layer PSR transformer, the 3-layer activity MLP, the Kendall caps — produces the *numerator*. Without the denominator, run11 at ep30 tells you a shared backbone reaches some absolute number, against which a reviewer's first question is "compared to what?" and you have no answer.

`scripts/train_st.py` already exists and is a unified `--task {det,act,psr,pose}` runner that builds the same MTL model and trains one head (verified, line 108). It is *finished code you have never launched.* This is not a "Path B is nice-to-have" situation. **The baselines are the paper.** 181 §4, 186 §2, and 192 §5 all said this. It is now the third round in a row where the mandatory experiment was deferred in favor of more architecture.

### Finding 2 — The efficiency claim, your paper's spine, has inverted.

Your own `efficiency_audit.md` (measured with fvcore, not estimated) establishes the ground truth:

| Quantity | Value | Source |
|---|---|---|
| 4 real specialists (YOLOv8m 25.9M + MViTv2-S act 34.5M + PSR 3–5M + MViTv2-S pose 34.5M) | **~100M** | efficiency_audit.md |
| Old MTL model (V8) | 53.8M | efficiency_audit.md (fvcore) |
| Old MTL saving | **~1.86–2.15×** | efficiency_audit.md |
| **run11 MTL model** | **117.7M** | 196 §1 |
| **run11 PSR head alone** | **70.9M** | 196 §4, verified in code |
| **run11 "saving"** | **0.85× (i.e. 18% LARGER than the specialists)** | this file |

You have converted a **2× parameter win into a 1.18× parameter *loss.*** The PSR head alone (70.9M) is **larger than the entire old MTL model (53.8M)** and **14–23× larger than the PSR specialist (3–5M).** A reviewer who computes params-per-task will see: "the multi-task model uses 70.9M for the step-recognition task, whose dedicated specialist uses 3–5M." That single line desk-rejects the efficiency contribution.

192 §1-Layer-4 spelled out the exact trap: *"the backbone swap does not trade some efficiency for accuracy; it deletes the paper's thesis."* You didn't swap the backbone — you kept MViTv2-S, correctly — but you achieved the identical inversion by ballooning a head. **The mechanism differs; the damage is the same.**

### Why these two findings dominate every Q in 200

Questions Q1–Q8 in file 200 are mostly downstream of these two findings. Q1 ("is one forward pass a defensible efficiency claim at 117.7M?") and Q2 ("PSR parameter explosion") are the same question: *shrink PSR or lose the paper.* Q5 ("ST baselines — now or later?") is *now, they gate everything.* Q3/Q7/Q8 are answerable only after the probe and baselines exist. So before the itemized answers, here is the corrected plan.

---

## 2. THE CORRECTED PLAN (Round-4 "Tier A-prime")

This is 192's Tier A, minus the drift that produced run11. Ordered by leverage, not by novelty.

**STEP 0 — Freeze the architecture right now.** No more heads, no more layers, no VideoMAE, no det-aug restart. Every architecture change since ep10 has been a reaction to a *training loss*, not to an *eval metric* — the exact anti-pattern (192 §7-1). Stop.

**STEP 1 — Run the overfit-200 eval probe. TODAY. (½ day, GPU-2.)** This is `overfit_50img_cls.py` — it exists, it has never been run. For each head: freeze backbone, overfit 50–200 fixed clips to ~0 train loss, then run the *real eval* on those same clips.
- If a head drives train loss →0 but eval metric stays ~0 → **the eval/target-encoding is broken**, and no architecture fixes it. This is the live hypothesis for detection's 0.0 (a prior ConvNeXt run hit 0.468, 176 §3.4) and a plausible one for activity's below-random.
- This probe is worth more than all of run11. It tells you whether you have been fixing a phantom.

**STEP 2 — Launch the 4 ST baselines. TODAY, in parallel. (`train_st.py`, GPU-2 staggered.)** They are the denominator. They are already coded. They take days, so they must start now, not "after run11." (See Q5 for the PSR-head-size caveat.)

**STEP 3 — Put the PSR head on a diet (see Q2 for the exact spec).** Target ≤15M. The task emits 8×11 = 88 numbers from an 8-token sequence; 70.9M is 5–7× oversized. If — and only if — the overfit probe shows PSR eval F1 actually moves, retrain the diet-PSR run. This restores the efficiency spine.

**STEP 4 — Only after Steps 1–3: let the (diet) MTL run reach ep30–50, then compute per-head MTL/ST ratios + CIs + the E8 gradient-cosine heatmap, and write L2+L3+method.**

**Cost:** the probe is ½ day; baselines run concurrently; the PSR diet is a config change + one retrain. This is *cheaper* than letting the 117.7M run11 grind to ep30 and *then* discovering you still have no baseline and a dead efficiency claim.

---

## 3. DIRECT ANSWERS TO THE FIVE "WHAT WE NEED FROM OPUS" ITEMS (200 §What-We-Need)

### (1) Primary recommendation: which path?

**Path B, gated on the overfit probe, with a mandatory PSR-head diet — NOT Path A, NOT Path C yet, NEVER Path D.**

- **Path A (run11→ep30→paper) is disqualified** on two independent grounds: it produces no ST baseline (so it cannot answer "helps vs hurts"), and it enshrines the 117.7M model that has already lost the efficiency argument.
- **Path B is the minimum publishable paper** and is mostly *already-coded work you keep deferring.* Do it.
- **Path C (soup) is a near-free increment on top of B** (192 Q5), *conditional* on all ST baselines using an identical backbone config — worth doing only after B's numbers are in, never as a reason to delay B.
- **Path D (scale backbone) remains rejected** (192 Q4). run11 is living proof that the bottleneck was never capacity — you 2.5×'d the model and activity is still below random.
- **Path E (per-head triage) is what you *think* you're doing but aren't** — triage requires the EP10 *eval* signal and the probe, which is Step 1, not more architecture.

### (2) EP10-contingent plan

The docs frame EP10 as the decision point. **It is not — the overfit probe is, and it's cheaper and faster.** But since EP10 will arrive, here is the contingency, and note that **every branch routes through the probe and the baselines, not through new architecture:**

| EP10 (diet-PSR or current) shows… | Action |
|---|---|
| Any head's eval metric ≈ 0 | Run the **overfit probe on that head first.** Loss-low + eval-zero = eval bug (fix the harness); loss-low + overfit-eval-high = real learning, just undertrained (wait). Do **not** add capacity either way. |
| PSR F1 > 0.05 | The P5 fix is real. Proceed to diet-PSR retrain to reclaim params. |
| PSR F1 < 0.02 despite low loss | **This is the focal-collapse-to-negatives case** (see Q6). Not a capacity problem. Fix: lower the positive/negative α imbalance, or evaluate at a swept threshold, or report event-F1 at the operating point that maximizes it. |
| Activity < 3% | **Do not add VideoMAE** (Q3). First rule out class-weight collapse: rerun eval with class weights *off* and with logit-adjustment *on* (already in the head, `logit_adjust=True`, currently disabled). Below-random is almost never a capacity symptom. |
| Detection < 0.01 | Overfit probe decides eval-bug vs feature-gap. If eval works on the overfit set, it's features/undertraining, not the head. |
| Pose ≤ 9° | Leave it alone. It's your positive-transfer candidate — but it only *becomes* positive transfer once ST-pose is measured (could be ~9° too, in which case it's "no cost," still fine). |

### (3) Parameter-budget verdict: is 117.7M defensible?

**No. 117.7M is indefensible as it stands, and 70.9M for PSR is the specific offense.** Not because 117.7M is a large number in the abstract — but because it is *larger than the baseline it claims to beat on efficiency* (~100M), and because the offending 70.9M sits on the task with the *smallest* specialist (3–5M). You must get the total **below ~55M** (roughly the old model) to keep a credible "≈2× fewer params than 4 specialists" line. That means PSR must drop from 70.9M to ≤~15M (Q2 gives the exact spec). The 3.75M activity head and 4.5M detection head are fine. The backbone (34.5M) is correct and must not change.

### (4) Paper framing (Q7's three options): which is strongest?

**Option 3 (positive transfer) is the *intended* headline but is contingent on data you don't have yet; Option 1 (Kendall pathology) is the *guaranteed* floor. Write the paper so that Option 1 is the load-bearing contribution and Option 3 is the empirical payoff — do not bet the paper on Option 3 alone, and drop Option 2.** Rationale in Q7 below. The one-line thesis:

> *"A single 34.5M-parameter MViTv2-S backbone with four lightweight task heads (total ≤55M) serves detection, activity, step-recognition, and ego-pose on IndustReal at ~2× the parameter efficiency of four specialists and single-pass latency. We identify and fix a degeneration in Kendall uncertainty weighting that otherwise starves the highest-loss task, and we report an honest per-task transfer map: positive transfer on pose, bounded cost on {…}, and a pre-registered honest miss on {…}."*

Note the blanks are filled by **measurement**, not by run11's loss curves.

### (5) ST-baseline timing

**Launch now, in parallel, on GPU-2. Do not wait for EP10 or for run11 to finish.** They are the long pole (days) and they gate the entire paper. One caveat: `train_st.py` builds the *same* MTL model and trains one head — so ST-PSR would inherit the 70.9M head. For the **accuracy ratio** that is the correct architecturally-matched control (isolates sharing). For the **efficiency narrative** you compare MTL against the *published* specialist sizes (~100M total). Report both comparisons and label them; do not conflate.

---

## 4. THE EIGHT SPECIFIC QUESTIONS (200 §Specific-Questions)

### Q1 — Is "one forward pass" a defensible efficiency claim at 117.7M?

**Not at 117.7M — the params are larger than the specialists, so "one pass" is all you'd have left, and it's not enough alone.** Latency/one-pass is a *real secondary* benefit, but it cannot carry the efficiency section when the headline parameter comparison is *underwater*. Reviewers read "efficient MTL" as "fewer parameters AND fewer passes." You currently have neither on params. **Fix the params (shrink PSR to ≤15M → total ≤~55M) and you get BOTH back:** ~2× fewer params *and* one pass *and* lower latency. Then "one forward pass for four tasks" is a clean, true, supporting point rather than a lonely fig leaf. Do not try to rescue 117.7M with the latency argument; rescue it by deleting ~55M of PSR.

### Q2 — The PSR parameter explosion: is there a functional-AND-defensible middle ground?

**Yes, and it's dramatic.** The head processes an **8-token sequence** (spatial-pooled P5, T=8) into **8×11 outputs.** d=768 with an **8× feedforward (6144)** across **6 layers** is wildly overparameterized for that. Verified culprits in code:
- `dim_feedforward = feat_dim * 8` (line 320) — **non-standard; the standard transformer FF is 4×.** The docstring literally says "[EP10] 4×→8× — bigger feedforward," i.e. it was doubled reactively. Revert to 4×: saves ~28M.
- d_model = 768 forced by reading P5 (768-dim) directly. Add a `Linear(768→256)` input projection and run the transformer at **d=256**. Attention cost scales d², FF scales d — this is the big lever.
- 6 layers (was bumped from 4). For an 8-token sequence, **2–3 layers** is ample.

**Recommended spec:** `Linear(768→256)` + **2-layer** causal Transformer, **d=256, nhead=4, ff=1024 (4×)**, → `Linear(256→11)`. That is **≈4–6M**, a **~12–15× reduction**, and for an 8-token/88-output problem it will very likely match or beat the 70.9M head (over-capacity on tiny sequences hurts as often as it helps). If you want to hedge, `d=384, 3 layers, ff=1536` lands ~11–13M. **Either restores the efficiency spine.** Retrain only after the overfit probe confirms PSR eval F1 actually moves — otherwise you're tuning a head whose signal you haven't verified.

One deeper point: **the P5-feature fix (96-dim conv_proj → 768-dim semantic) is the change that mattered, and it is orthogonal to head size.** 192 FC-4 and 186 B-6 both located the bottleneck at the *feature source*, not the decoder. You can keep the good feature source and shrink the decoder by 15×. Do that.

### Q3 — Activity: keep the 3-layer MLP, or add VideoMAE?

**Neither is your problem. The 3-layer MLP already over-invested in the wrong place, and VideoMAE would compound it.** The decisive fact (186 §0, 192 Q3): **the 0.6525 activity SOTA was set by MViTv2-S + a single linear layer + plain CE.** A linear head suffices to hit SOTA single-task, so *the head is not the bottleneck* — the shared representation is. Your 3-layer MLP is treating a disease the patient doesn't have.

The 0.58% below-random tells you the real disease, and it isn't capacity — a random head scores 1.33%, so **0.58% is worse than random**, which only happens when the model *systematically* predicts classes that are rare in the test set. Verified mechanism: training applies `compute_activity_class_weights` (sqrt-tamed inverse-frequency, up to 11.71×, line 299/365). A cold or lightly-trained head under heavy rare-class upweighting collapses onto rare classes → below-random top-1. **This is a loss-weighting/label pathology, not a representation-capacity ceiling.**

**Action, in order (all cheap):** (a) run the overfit probe on activity — if it overfits 200 clips to high top-1, backbone+head are fine and the deficit is MTL+weighting; (b) rerun eval with class weights **off** (weighted CE is for training gradient balance, not a reason for the argmax to prefer rare classes — but if it has collapsed, unweighted eval exposes it); (c) enable the **logit-adjustment** already sitting unused in `ActivityHead` (`logit_adjust=True`, Menon et al. 2020) — the principled long-tail fix 192 Q3 named. **VideoMAE only enters the conversation if, after ST-activity establishes the ceiling, MTL activity provably cannot reach it — and even then it costs you the efficiency claim (+22M) that you're already fighting to keep.** There is no activity threshold at which VideoMAE is the right first move. Trust the representation; fix the weighting.

### Q4 — Detection augmentation (`--det-aug`) now or after EP10?

**After the overfit probe, not now, and not "because EP10 was low."** 192 Q6 endorsed mosaic/mixup *for detection specifically* (detection is the genuinely data-limited head) — so det-aug is a legitimate lever *in principle.* But turning it on now, before you know whether detection's 0.0 is an **eval bug** (live hypothesis, per the 0.468 prior run) or a **feature/assigner gap**, is another blind architecture change. If the probe shows the eval harness is broken, det-aug does nothing but add noise and a restart. Sequence: probe → if features/data are the real gap → then det-aug. Don't restart training for it speculatively.

### Q5 — ST baselines: parallel, or after run11?

**Parallel, starting now** (covered in §3-item-5). The GPU-contention worry is minor — stagger them; even sequential they must start today because they're the long pole. The real subtlety you raised is correct and important: **what head does ST-PSR use?** Answer: for the *matched-architecture accuracy control*, ST-PSR uses the **same (diet) PSR head** as MTL — that's the clean "sharing vs not" comparison. For the *efficiency* table, you cite the **published specialist** (~3–5M). Keep these two comparisons in separate columns and label them; a reviewer who sees a 70.9M "PSR specialist" will (correctly) object, which is one more reason to do the diet *before* the baselines so both MTL and ST-PSR use the ≤15M head.

### Q6 — The "honest miss" story: does PSR's apparent strength change the narrative?

**Be very careful here — the docs are over-reading the loss drop, and the "surprising strength" framing is premature.** Under **focal-BCE (α=0.25, γ=2) on transitions that are <1% of frames**, a model that predicts "no transition everywhere" earns a *very low* loss, because focal down-weights the abundant easy negatives and there are almost no positives to penalize. **0.17 focal-BCE is consistent with both real learning AND total collapse-to-negative — and the latter yields event-F1 = 0.** 197 §5 and 198 §3.3 both gesture at this ("loss being low doesn't guarantee the right predictions… pessimistic: always 'no transition'"). So: **do not narrate PSR as a "breakthrough" or "surprising strength" until event-F1 on real eval is non-zero.** The loss drop proves the *feature source* was the bottleneck (real, worth a sentence); it does not prove the task is solved. **Keep PSR pre-registered as the honest miss** (186 G-2, 192 Q7). If eval F1 surprises to the upside, upgrade the narrative *then*, with the metric in hand. Meanwhile the actual "miss" candidates are detection and activity, and which one it is will be decided by the probe + baselines — not now.

### Q7 — Minimum viable paper: which of the three options?

**Ranking: Option 1 (Kendall pathology) is the guaranteed floor and should be the methodological spine; Option 3 (positive transfer) is the empirical headline *if the data delivers it*; Option 2 (efficiency trade-off) is the weakest — do not lead with it.**

- **Option 1 (Kendall-collapse characterization + fix)** is the one contribution that is **already true regardless of any eval number.** You have the mechanism (inverse-loss-scaling starvation), the fix (capped precision + EMA normalization), and — critically — you can *demonstrate the collapse* by an ablation: uncapped Kendall vs capped, showing the highest-loss head gets starved to zero without the cap. That ablation is worth running explicitly (it's your Figure-1 alongside the E8 gradient heatmap). The reviewer risk ("you just tuned hyperparameters") is real and is answered *by the ablation*, not by prose: show uncapped weighting drives a head's effective weight → 0 and its metric → chance, and capped weighting rescues it. That's methodology, not tuning.
- **Option 3 (positive transfer)** is the *headline you want* and is likely true for pose — but it is **contingent on ST-pose actually being worse than MTL-pose**, which you have not measured. If ST-pose is also ~9°, Option 3 softens to "no cost on pose," which is still fine but not a headline. **Build the paper so it survives that outcome:** lead with Option 1's method, report Option 3's transfer map as the empirical result, and let the strongest measured head carry the "helps" claim.
- **Option 2 (efficiency trade-off)** is both the least novel ("everyone knows MTL trades accuracy for speed") *and* the one run11 has actively undermined (you're not currently more efficient). Demote it to a supporting section, and only after the PSR diet restores the real ~2× number.

**So the strongest paper if the ideal doesn't materialize = Option 1 (method) as spine + honest per-head transfer map (the good part of Option 3) + restored efficiency (the salvaged part of Option 2).** That is precisely the L2+L3+method paper 181/186/192 have pointed at for three rounds. It does not require beating any SOTA. It does require the baselines and the probe.

### Q8 — The MViTv2-S ceiling: what's the theoretical MTL ceiling?

**There is no clean closed-form ceiling, but the right mental model corrects your framing in your favor — and it is an argument *for* keeping MViTv2-S, not for scaling.** A shared backbone *can* in principle represent everything a dedicated one can (the parameters are a superset of a single-task init), so the loss is not information-theoretic inevitability — it is an *optimization* phenomenon: gradient interference and a finite-capacity bottleneck (the single 768-dim class token for activity+pose). That's why the honest ceiling is **empirical, measured by your ST baselines**, and why the *ratio* MTL/ST is far more robust and more publishable than any absolute. Your instinct in Q8 is right: **report MTL as a fraction of the *matched ST ceiling you measure*, not as a fraction of a different system's published SOTA** (192 FC-6 made the same move for detection: judge against IndustReal-only 0.779, not synthetic-augmented 0.838). If ST-activity on your data/protocol lands at, say, 0.45–0.55 (plausible for clip-level MViTv2-S under your exact eval), then MTL at 0.40 is 75–90% of the *reachable* ceiling — a strong bounded-cost story — even though it's only ~60% of the 0.65 headline SOTA. **Measure the ceiling; stop benchmarking against numbers set by other systems on other data.**

---

## 5. WHAT RUN11 GOT RIGHT (so this isn't all correction)

Credit where due, verified in code:
- **PSR feature source (conv_proj 96-dim → P5 768-dim)** is the correct fix and the loss drop confirms the *diagnosis* (192 FC-4, 186 B-6). Keep the feature source; shrink the decoder.
- **PSR native-T=8 prediction** (no more 8→16 linear interpolation) is correctly implemented (`PSRHead.forward`, labels max-pooled T=16→8 in `psr_loss`). Good — this was a real bug and it's fixed.
- **Detection P2 skip** (dropping raw conv_proj from the detection heads) is implemented and correct (192 FC-2/Layer-5).
- **TAL assigner + DFL/CIoU/focal** is a legitimate, properly-cited (TOOD/GFL/YOLOX) upgrade over sparse 3×3 — the *right* second-order lever (192 Q1). The alternating 0.001/4-5 loss is expected behavior, not a bug.
- **Decoupled detection head** already existed and is retained.
- **Kendall caps + EMA normalization + PCGrad** are implemented and are your methodological contribution.

The problem was never that these are wrong. The problem is that they were shipped **without the eval verification and baselines that tell you whether they worked**, and bundled with a **PSR head 15× too large** that mugged the efficiency claim.

---

## 6. ONE-SCREEN SUMMARY

```
195–200 ASSUME                          →  ROUND-4 VERDICT (code + measurements)
────────────────────────────────────────────────────────────────────────────
"run11 losses look better ⇒ progress"   →  Loss ≠ eval ≠ baseline. Below-random activity
                                           = weight/label collapse; PSR 1.56→0.17 under focal
                                           = consistent with F1=0. Prove with the eval, not curves.
"117.7M is defensible, latency saves it" →  NO. Specialists ≈100M; run11 is LARGER. Efficiency
                                           claim (the paper's spine) is INVERTED, exactly the
                                           foundation-backbone trap 192 named — via a bloated head.
"PSR 70.9M is the price of features"     →  FALSE. Feature source (P5) ≠ decoder size. 8 tokens →
                                           88 outputs needs ≤15M. ff is 8× (should be 4×), d=768
                                           (use 256), 6 layers (use 2-3). Cut to ~5M, keep P5.
"3-layer MLP fixes activity"             →  NO. SOTA = 1 linear + CE. Head isn't the bottleneck.
                                           0.58%<random = class-weight collapse. Use the
                                           logit-adjust already in the head; don't add VideoMAE.
"Path A: run11→ep30→paper"               →  DISQUALIFIED. No baseline = can't answer helps/hurts;
                                           enshrines the dead efficiency claim.
"EP10 is the decision point"             →  NO. The overfit-200 probe is — and it's coded, ½ day,
                                           NEVER RUN. Neither were the 4 ST baselines (train_st.py).
────────────────────────────────────────────────────────────────────────────
DO INSTEAD (Tier A-prime):
  0. FREEZE architecture. Stop reacting to training loss.
  1. Run overfit_50img_cls.py TODAY (½ day) — is each head's EVAL metric alive?
  2. Launch 4 ST baselines TODAY, parallel (train_st.py --task) — they ARE the paper.
  3. PSR head diet 70.9M → ≤15M (Linear 768→256, 2-3 layers, d=256, ff 4×). Restore ~2× efficiency.
  4. THEN diet-MTL → ep30-50 → per-head MTL/ST ratios+CIs + E8 heatmap + Kendall-collapse ablation.
  5. Write L2(transfer map) + L3(restored efficiency) + method(Kendall fix). No SOTA-chase.
PAPER SPINE: Kendall-collapse method (always true) + measured transfer map + ~2× efficiency.
FIRST MOVE: the overfit probe. You've built four diagnostic scripts and run none of them.
```

*Companion to 181 (round 1), 186 (round 2), 192 (round 3). Same authority rule: where this file and 195–200 disagree, this file is grounded in the executed code and the fvcore measurements, and supersedes them. The short version: you asked "can we prove MTL helps?" — and then, for the third round running, built architecture instead of running the one comparison that would answer it, while inflating a head until the efficiency thesis you already had reversed sign. Freeze the model. Run the probe you wrote. Run the baselines you coded. Shrink PSR back under the specialist total. The strong, honest, reviewer-proof paper — Kendall-collapse fix + measured per-task transfer + genuine ~2× efficiency — is the one your data already supports, and it is smaller, not bigger, than what you have now.*
