# 207 — Opus Round 5 Answer: Architecture Strategy Verification — The Repo Contradicts the Prompt

**Date:** 2026-07-10
**Inputs read line-by-line:** 202 (SOTA path), 203 (implementation specs), 204 (training methodology), 205 (risks/timeline), 206 (this round's prompt). Context re-read: 201 (Round 4 answer), 195–200 (Round 4 prompt), efficiency_audit.md.
**Ground truth re-verified against the executed code this round:**
`src/models/mvit_mtl_model.py` (PSRHead :297–363, ActivityHead :219–290, PoseHead :370–379, head instantiation :436–445, pose renorm :521–524), `scripts/train_mtl_mvit.py` (Kendall caps :777–786, EMA normalization :740–764, PSR focal :375–413, CLI flags :1352–1386), `scripts/train_st.py` (compiles at HEAD, `--task` runner), `scripts/overfit_50img_cls.py` (compiles at HEAD), git history since Round 4 (`git log --name-only --since=2026-07-09`), full-repo searches for `kendall_uncapped`, `logit_adjust=True`, `input_proj`, `run12`, `overfit_probe.py`.
**Authority rule (unchanged from 181/186/192/201):** where documents 202–206 and the code/measurements disagree, the code and measurements win, and I say so explicitly. This round, that rule does most of the work.

---

## 0. READ THIS FIRST — the one-paragraph answer

**Round 5 asks me to verify an architecture strategy, but the repository does not contain the work the strategy is predicated on.** File 206 opens with five completed items — PSR diet, activity logit-adjust enabled, `--kendall-uncapped` wired, `overfit_probe.py` rewritten, `train_st.py` fixed, run12 live at 48.6M. **None of the five is in the pushed code** (verified exhaustively in §1: the committed `PSRHead` is still the 6-layer/d=768/ff=8× 70.9M head; `ActivityHead` is instantiated *without* `logit_adjust`; the string `kendall_uncapped` appears only in docs 204–206; `scripts/overfit_probe.py` does not exist; the only commits since my Round 4 answer are documentation). The EP10 section of 206 is still `[PENDING]` and no run12 artifact — log, checkpoint, eval JSON — exists anywhere in the repo. So the honest Round 5 verdict has to start here: **either the run12 work exists on the local GPU machine and was never pushed (then push it today — an unpushed experiment is an unverifiable experiment under this series' own authority rule), or the documents once again describe intentions as completions**, the exact failure mode your own `efficiency_audit.md` documents for files 167/170 and your own 205 Risk 1 names "REPEAT OFFENDER." On the actual questions: the 3-lever strategy of 202 is **one good conditional lever (detection neck), one factually-broken lever (the VideoMAE case rests on a wrong benchmark number — ViT-B is ~81.5% on K400, not 87.4%), and one self-defeating lever (per-task LoRA on backbone Q/V forces four backbone passes and thereby deletes the single-forward-pass claim — the last intact half of your efficiency story).** Trust the 1.78M PSR diet (its arithmetic is right; capacity is not PSR's failure mode); do not build the 5.2M detection-conditioned head — the spec in 203 §2.1 contains a confirmed bug that makes per-frame transition localization impossible (mean-pool-then-expand returns identical logits for all 8 frames). Skip Nash-MTL for the paper (204's pseudocode is not Nash-MTL, and swapping the method mid-paper deletes your Kendall-caps contribution). Skip the SlowFast-Hydra-MTAN rewrite (it starts −4.86 top-1 on your best-anchored task). **The one thing to do tomorrow: push the real run12 code state, then run the overfit probe — the experiment that has now been "the next thing to do" for five consecutive rounds — while ST-pose launches on GPU 2.** Everything in 202–205 is a menu to order from *after* the probe and EP10 tell you what is actually broken.

---

## 1. FINDING 1 — The repository does not contain Round 4's claimed work

File 206 §"What We Did Since Round 4" lists five completions. I verified each against the repo at HEAD (`cb974ef25`):

| # | 206 claims | What the repo actually contains | Verdict |
|---|---|---|---|
| 1 | "PSR diet: 70.9M → 1.78M (Linear 768→256 proj + 2-layer T d=256 ff=4×). run12 launched at 48.6M" | `PSRHead` at `mvit_mtl_model.py:297–326` is still **6 layers** (`:312`), **ff = feat_dim × 8** (`:320`), no input projection, instantiated with `feat_dim=backbone_dim` = 768 (`:445`) → **70.9M, the run11 head**. No `Linear(768→256)` exists anywhere in the model file. | **ABSENT** |
| 2 | "Activity logit-adjust: Enabled Menon et al. 2020" | `ActivityHead` *supports* `logit_adjust` (default `False`, `:242`) but is instantiated **without it** (`:436`: `ActivityHead(feat_dim=backbone_dim, num_classes=num_act_classes)`). No `--logit-adjust` flag in `train_mtl_mvit.py` (checked the full `add_argument` list, `:1352–1386`). | **ABSENT** |
| 3 | "Kendall ablation: `--kendall-uncapped` flag wired" | Full-repo grep for `kendall.uncapped\|kendall_uncapped`: **3 hits, all in docs 204/205/206. Zero in code.** | **ABSENT** |
| 4 | "Overfit probe: `scripts/overfit_probe.py` rewritten for current MViTv2-S arch" | **The file does not exist.** Only `overfit_50img_cls.py` (Tier A era, commit `c6d2f5259`) exists; it compiles, but it is not the claimed rewrite. | **ABSENT** |
| 5 | "ST baseline fix: `scripts/train_st.py` syntax bugs fixed" | `train_st.py` compiles cleanly at HEAD — but its last touching commit is `f2b01cc4a` (Tier A, pre-run11). No fix commit exists. Either it never had the claimed syntax bugs, or the fix was never pushed. | **UNVERIFIABLE** |
| — | "run12 LIVE at epoch 10 eval (48.6M diet model)" | No `runs/` artifact, no log, no eval JSON references run12. The repo-root `eval_results.json` is a stale `epoch_0_batch_200` snapshot (activity 2.2%, pose 30°). 206's EP10 section is `[PENDING]`. | **UNVERIFIABLE** |

The complete commit history since my Round 4 answer: `9c523c70a` (docs 195–200), `fb3740936`/`a1c144d42` (doc 201), `595789bd5` (merge), `05e7bf2d2` (docs 202–205), `cb974ef25` (doc 206). **The last commit that touched executable code is `d057a1688` — the run11 head upgrades, *before* Round 4.**

**What this means.** There are two possible worlds. (a) The diet/logit-adjust/flag/probe work exists in the local working tree of the GPU machine and was never committed — in which case run12's 48.6M arithmetic is at least internally consistent (117.7M − 70.9M + 1.78M ≈ 48.6M ✓, and 100M/48.6M ≈ 2.06× ✓), and the fix costs 30 minutes: `git add`, commit, push, plus the run12 launch command and log into the repo. (b) The work was described but not done. **I cannot distinguish (a) from (b) from here, and neither can a reviewer, a collaborator, or you in three weeks.** This project has already had one documented incident of numbers existing only in prose (`efficiency_audit.md`: the fabricated 600M/6.7× tables in 167/170). The rule that fixes both worlds is the same: **from now on, a consultation round may not open until the repo contains (1) the code diff and (2) at least one result artifact from the previous round's mandatory experiments. If it isn't pushed, it didn't happen.**

### FINDING 1b — The Kendall caps in the code are not the caps in the docs, and detection is the least-protected head

Docs 196 §7.2 and 198 §5.2 state the caps as `act≤1.0, psr≤0.5, det≤1.5, pose≤2.0` (weight floors 0.37/0.61/0.22/0.14). The executed code says otherwise (`train_mtl_mvit.py:778`):

```python
LV_CLAMP_MAX = {"det": 4.0, "act": 1.0, "psr": 0.5, "pose": 4.0}
```

Since the effective weight is `exp(-log_var)`, a cap of 4.0 permits a weight floor of **exp(−4) ≈ 0.018** — detection and pose can be down-weighted ~12× further than the documents believe. Pose is additionally tied to detection's precision (`hp_prec_cap`, `:783–786`), which is fine. But **detection — the head that has shown 0.000 mAP in every eval — has the weakest starvation protection in the entire system.** The EMA normalization (`:759–764`) mitigates this by feeding Kendall O(1) losses, so the caps rarely bind; but "rarely" is not "never," and a drifting `log_var_det` toward 4.0 is a cheap, checkable starvation signature. **Add to the EP10 checklist: log the four `log_var` trajectories. If `log_var_det` is pinned near its cap while detection is at 0, the Kendall-collapse story you plan to publish is happening in your own training run, un-diagnosed.** This is simultaneously a risk and — properly instrumented — free evidence for your Figure 1.

---

## 2. FACT-CHECK — What survived validation of the 22-agent research

You asked for verification. I checked every load-bearing claim in 202–204 against the code, hand-computed arithmetic, and the primary literature. Findings, most consequential first:

### 2.1 CORRECTED — VideoMAE ViT-B does not score 87.4% on K400 (kills Lever 1's headline)

202 §1.1 lists "VideoMAE ViT-B: 86M params, 87.4% K400 top-1" and derives "+6.4% absolute gain from better pretraining" (§1.2). **The published VideoMAE (Tong et al., NeurIPS 2022) ViT-B result on Kinetics-400 is ≈81.5% top-1; 87.4% is not a ViT-B number** (VideoMAE ViT-L is ≈85.2, ViT-H ≈86.6; 87+ belongs to much larger or later models such as VideoMAE V2 giants). Similarly, the "75.4% SSv2" cited for ViT-B is the **ViT-L** figure; ViT-B lands ≈70.8%. The honest comparison is: **MViTv2-S 81.0 vs VideoMAE ViT-B ~81.5 — a ~0.5-point difference for 2.5× the parameters (86M vs 34.5M),** which would single-handedly re-invert the efficiency claim (86M backbone + heads > 100M specialists) and break the apples-to-apples WACV comparison (the 65.25% IndustReal SOTA is MViTv2-S). 205 already put the VideoMAE swap in "SKIP" — correct — but 202 presents it as Lever 1 of 3 and 206 Q1 repeats it. **Lever 1 is rejected, and now on factual grounds, not just strategic ones.**

### 2.2 CORRECTED — Per-task LoRA adapters contradict the single-forward-pass claim (kills Lever 3 as specced)

This is the most important architecture-level finding of the round. LoRA on the backbone's Q/V projections means **the backbone's activations are task-dependent from the first adapted block onward.** `TaskAdapterStore.set_task()` (203 §5.3) makes this explicit: one task per forward. Serving 4 tasks therefore requires **four backbone passes** (or a batch-of-4-duplicates trick that costs the same compute). Your paper's efficiency claim, after the PSR diet rescued the parameter half, rests on exactly one sentence: *"one forward pass serves four tasks."* Per-task backbone adapters delete that sentence. **This is the same inversion as the 70.9M PSR head and the foundation backbone before it — a per-task mechanism smuggled into the shared component, discovered each time by checking the claim it breaks.**

Secondary but worth recording:
- **The adapter parameter math in 202/203 is wrong twice over.** 202 §3.2 assumes "24 layers × 768-dim"; MViTv2-S has **16 blocks** (depths [1,2,11,2]) at **stage dims 96/192/384/768** — not uniform 768. Recomputing LoRA r=8 on Q/V at true dims: 32r·Σd = 32·8·(96·1+192·2+384·11+768·2)/8… = **≈0.20M per task**, and FiLM = 2·Σd ≈ **12.5K per task** — so the true total for 4 tasks is **≈0.85M, not 4.2M**. (203's own premise of 16×768 blocks still yields 393K/task, not the stated 786K; the FiLM line "16×2×768 = 197K" actually equals 24.6K — off 8×.) The error is in your favor on params but confirms nobody checked the arithmetic.
- **The mechanism's premise self-cancels in 204's own schedule.** Adapters "eliminate gradient interference because the backbone is frozen" (202 §3.3) — true only in Phase 1. Phase 2 (204 §1.2) unfreezes the backbone for 45 epochs of joint training, at which point interference returns and the adapters are just extra capacity. What Phase 1 actually buys is a warm start — legitimate, but a much smaller claim than "eliminates gradient interference."

### 2.3 CORRECTED — 203's DetectionConditionedPSRHead cannot localize transitions (confirmed bug in the spec)

203 §2.1, final line of `forward()`:

```python
x = x.mean(dim=1)  # [B, 256]
return self.classifier(x).unsqueeze(1).expand(-1, 8, -1)  # [B, 8, 11]
```

Mean-pool over time, classify once, then **copy the identical logits to all 8 frames.** Every frame in a window receives the same prediction, so within-window 0→1 transitions — the events `event_F1@±3` scores — can only appear at window boundaries, quantized to the sliding-window stride. The current committed head (and the claimed diet head) predicts **per-frame** logits from per-frame tokens, which is the correct shape for this metric. If you ever build a detection-conditioned PSR head, the classifier must be applied per-frame (`Linear` on `[B, T, d]`), not on the pooled vector. As written, the 5.2M head is a downgrade wearing an upgrade's params.

### 2.4 CORRECTED — 204's Nash-MTL pseudocode is not Nash-MTL

204 §1.2 computes `alpha = solve(G, ones)` (G = Gram matrix of task gradients). **Navon et al. (ICML 2022) define the Nash bargaining solution by the fixed-point condition `GᵀG α = 1/α` (element-wise reciprocal), solved via a sequence of concave approximations** — there is no closed-form linear solve, and `Gα = 1` is a different (unnamed) scheme with none of the cited guarantees. Also unpriced in 204: Nash-MTL needs per-task backbone gradients every step (K=4 backward passes, same cost class as PCGrad's surgery) and is noisy at your effective batch size of 4 — the Gram matrix of four minibatch gradients at batch 4 is mostly noise, which is why the paper itself updates α at intervals. **If you ever run it, use the published algorithm, not 204's.** (My recommendation in §5 is that you don't, for the paper's sake.)

### 2.5 CORRECTED — smaller but real spec defects

- **BiFPN P5 fusion duplicates a term** (203 §1.1): `w[0]*p5_lat + w[1]*max_pool(p4_out) + w[2]*p5_lat` — `p5_lat` enters twice; the comment says the third input should be the top-down P5, which doesn't exist as coded. Two-input fusion or a genuine third input — pick one before implementing.
- **BatchNorm at batch size 2** (203 §1.2 GFLV2 head): BN statistics at batch 2 (your VRAM-limited setting) are noise. Detection heads in this regime use **GroupNorm** (FCOS/ATSS/GFL practice). One-line change, prevents a silent accuracy tax.
- **The "GFLV2" quality head is a variant, not GFLV2**: DGQP in the paper ingests the **top-k values (k=4) of each edge's DFL distribution**; 203 uses mean/std. Probably fine, but don't cite it as GFLV2 verbatim in the paper.
- **Budget inconsistency:** 202/206 claim "~8M BiFPN/det branch" and a ~60M total; 203's own specs sum to BiFPN 1.2M + det head ~0.36M + PSR 5.2M + act 5.0M + pose 0.5M + adapters ~0.85M (corrected) + backbone 34.5M ≈ **47.6M** — the full spec is actually *cheaper* than the budget doc says. Nobody cross-checked 202 against 203.
- **Doc/code drift on the pose head:** 196/202 list pose at 0.5M; the committed head (`Linear(768→256→6)`, `:373–379`) is **≈0.20M**. Cosmetic, but it's a third instance of numbers not tied to code.

### 2.6 CONFIRMED — the things that held up

| Claim | Verification | Verdict |
|---|---|---|
| Diet PSR = 1.78M at d=256/2L/ff=1024 + `Linear(768→256)` + `Linear(256→11)` | Hand-computed: per layer ≈ 789.8K ×2 + 196.9K proj + 2.8K cls ≈ **1.78M** | ✅ (as arithmetic; not yet as code) |
| run12 total 48.6M, 2.06× vs ~100M specialists | 117.7 − 70.9 + 1.78 ≈ 48.6; 100/48.6 = 2.06 | ✅ arithmetic (specialists ~100M re-confirmed in `efficiency_audit.md`, fvcore) |
| MViTv2-S 34.5M / 81.0 K400; IndustReal AR SOTA 65.25 with MViTv2-S + linear head | MViTv2 paper; IndustReal WACV paper (internal docs consistent) | ✅ |
| 6D rotation + Gram-Schmidt (Zhou et al., CVPR 2019) superior to quaternion regression; geodesic loss is the right SO(3) metric | Primary source | ✅ — with one integration caveat: **your GT is fwd/up vectors, not rotation matrices** (`mvit_mtl_model.py:521–524`), so build `R_gt` by orthonormalizing (fwd, up) before the geodesic — and report fwd/up MAE alongside, for continuity with prior epochs |
| Decoupled training (Kang et al., ICLR 2020) — classifier retrain with class-balanced sampling, +2–10% on long-tail | Primary source; gains of that order on ImageNet-LT | ✅ cheap and appropriate for the 75-class tail |
| Logit adjustment (Menon et al., ICLR 2021) as the principled long-tail fix | Primary source | ✅ — but see the **wiring caveat** below |
| TAL (TOOD, ICCV 2021) with α=1, β=6; DFL/QFL (GFL, NeurIPS 2020) | Primary sources; committed assigner matches (`train_mtl_mvit.py:168`) | ✅ (per-level top-k 9/12/15 is your own tweak — fine, label it as such) |
| Mosaic/CopyPaste help small detection datasets | YOLOv4 lineage + Ghiasi et al. 2021 | ✅ directionally; "+3–5 mAP" is an estimate, not a citation |
| SlowFast on IndustReal = 60.39 top-1 < MViTv2-S 65.25 | Internal benchmark table (198 §2.1) | ✅ — this is the number that sinks Q5's rewrite |
| ASFormer (BMVC 2021) is an encoder-decoder for **untrimmed, minutes-long** sequences (thousands of frames, dilated temporal attention) | Primary source | ✅ — which is exactly why it's the wrong regime for your T=8 windows (§4) |

**Logit-adjust wiring caveat (check before enabling in run13):** the committed `ActivityHead.forward` applies `logits + τ·log(class_freq)` **unconditionally — in eval as well as training** (`:285–289`). Menon et al.'s additive formulation puts the adjustment **inside the training loss** and predicts from **raw logits** at test time; applying the same additive term at eval reproduces the ordinary (imbalanced) maximum-likelihood prediction and neutralizes the debiasing you enabled it for. Fix: apply the adjustment in `activity_loss` (or gate on `self.training`), and evaluate on raw logits. If run12 "enabled logit-adjust" through this forward as-is, expect it to look like it did nothing at EP10 — *that would be the wiring, not the method.*

**A general note for the paper:** every "expected gain" figure in 202–205 (+5-8% temporal pool, +0.05-0.15 F1 det-conditioning, +2-8% Nash-MTL, 93-97% distillation) is an *extrapolated estimate from other benchmarks*, not a measured or even cited-verbatim number. That is fine for planning documents. It is not fine if any of them migrates into the paper without a measurement behind it — this project has been burned by exactly that once already.

---

## 3. Q1 — Is the 3-lever strategy correct?

**No as a package. One lever survives, conditionally.** And your own framing of the alternative — "a problem whose real bottleneck is data volume and evaluation bugs we haven't ruled out" — is the correct null hypothesis that 202 never tests.

- **Lever (a), backbone pretraining swap: REJECT.** The case was built on a wrong number (§2.1). At true published numbers the swap buys ~0.5 points of K400 pretraining quality for 2.5× backbone params, breaks the MViTv2-S-vs-MViTv2-S comparison with the WACV SOTA, and re-inverts the efficiency claim. 205 already had this in SKIP; 202/206 should be corrected to match. The *Ego4D domain-adaptation* idea is independently reasonable but is a Tier-3 luxury for a later paper.
- **Lever (b), detection-specialized neck: KEEP, GATED.** BiFPN@128 + decoupled 96ch head is cheap (~1.6M by 203's own specs, *less* than the current 256ch FPN+head at ~6M), evidence-based, and doesn't disturb other heads. But it is the *second* move for detection. The first is knowing why mAP is 0.000: the overfit probe distinguishes eval-harness failure from feature/assigner failure, and `--det-aug`/mosaic (already implemented, never activated — `train_mtl_mvit.py:1363`) is the zero-param lever with the largest published upside on 27K annotated frames. Sequence: probe → augmentation → BiFPN (with the §2.5 fixes) → GFLV2 quality branch last, and only if mAP is already >0.3 and you're hunting decimals.
- **Lever (c), per-task adapter isolation: REJECT as specced.** It deletes the single-pass claim (§2.2), its parameter accounting is wrong, and its interference-elimination premise only holds while the backbone is frozen — which your own Phase 2 undoes. If ST baselines later prove a large, persistent MTL gap on one head, the *salvageable* version is **task-conditional adapters in the final stage only** (stage 4 = 2 blocks): 14 of 16 blocks stay a genuinely shared single pass, and the per-task tail costs ~4× of two blocks ≈ a few percent latency. That is a contingency, not a plan.

**The direct answer to your question:** yes, you are overcomplicating. The evidence you already have — PSR loss responding 10× to a *feature-routing* fix; activity below random (a label/weighting signature, not a capacity one); detection at exactly 0.000 with an unrun probe; 78K windows against a then-117.7M model — points at eval bugs, loss/imbalance handling, and data volume, in that order, before architecture. 202–205's own Tier 1 list agrees: 3 of its first 5 items (ST baselines, probe, Kendall ablation) are *experiments*, not architecture. Run them.

---

## 4. Q2 — Is the PSR head diet sufficient?

**Trust the 1.78M head. Do not build the 5.2M detection-conditioned head now. ASFormer does not conflict — it's the wrong regime entirely.**

1. **The diet is right-sized, not underpowered.** The task maps 8 tokens → 88 logits. A 2-layer d=256 ff=1024 transformer has ~1.6M params of sequence modeling for a length-8 sequence — this is generous by any standard (BERT allocates ~7M/layer for length-512). The load-bearing fix was the **feature source** (conv_proj 96-dim → P5 768-dim; the 1.56 → 0.17 loss drop happened at d=768 *and* survives at d=256 per your reported 0.27), and the diet keeps that fix while restoring the efficiency spine. The "0.27 vs 0.17 training-loss parity" is weak evidence in isolation — under focal-BCE with <1% positives both numbers are consistent with collapse-to-negatives (201 Q6 stands) — but that cuts against the 70.9M head equally. **Capacity is not the axis PSR fails on; the operating point is.** Before any PSR architecture change: (a) run the overfit probe on PSR; (b) sweep the binarization threshold (eval hardcodes 0.5) and report event-F1 at the best operating point; (c) if positives are being ignored, apply the transition-aware weighting from 203 §2.2 — a loss change, zero params.
2. **The 5.2M detection-conditioned head: not now, and never as specced.** Three independent reasons: (i) the spec has the confirmed localization bug (§2.3); (ii) it conditions PSR on a detection stream currently emitting 0.000 mAP — gating on `sigmoid(0)=0.5` means injecting 50%-weighted noise at initialization; the gate can learn to close, but you'd be spending params to feed a head garbage it must learn to ignore; (iii) the causal story. Make detection-conditioning a *contingency* with an explicit gate: **only if detection mAP > 0.3 and PSR F1 is alive-but-plateaued**, and rewrite the classifier to be per-frame first.
3. **Bi-directional attention is a legitimate, free upgrade — decide it as protocol, not architecture.** The committed head is causal (`:355–360`) to support an online story. Your PSR *metric* is offline per-recording, and STORM is offline too. Dropping the mask is one line, costs nothing, and plausibly helps ("a transition is confirmed by what happens after it"). But it forfeits the "online-capable" sentence. Pick one, document it in the eval protocol, and don't flip-flop between runs — it changes what F1 means.
4. **ASFormer: no conflict, no relevance at T=8.** ASFormer's contributions (dilated temporal attention, iterative refinement decoder) exist to handle sequences of thousands of frames in untrimmed videos. On an 8-token window, its attention pattern degenerates to full attention — which is what you already have. The two-stream agent recommended an architecture for a *different problem shape* (full-video segmentation) than your windowed per-frame formulation. The one idea worth stealing later, if PSR is alive but noisy, is the *refinement* pattern: a second pass that smooths per-frame logits across windows at inference (even a 1-D median filter over stitched logits is the poor-man's version and costs zero params). File it under post-processing, not architecture.

---

## 5. Q3 — Is adapter-based MTL + Nash-MTL the right mechanism?

**No — and your own instinct in the question ("does this complicate the paper's story?") is the right one; the technical audit just makes it decisive.**

- **The efficiency contradiction is disqualifying** (§2.2). "Shared backbone + per-task backbone adapters" is an oxymoron at inference time: four passes, or one pass per task with batching tricks that cost the same FLOPs. You would be publishing "one backbone, four tasks, ~2× fewer parameters, single forward pass" with a mechanism that makes the fourth clause false. Reviewers who missed the 70.9M PSR head would not miss this.
- **The gain is speculative; the cost to the story is certain.** "+3-8% per task" is a transfer estimate from VTAB-style single-task PETL benchmarks, not an MTL measurement on anything resembling your setup. Meanwhile the paper's methodological identity — *Kendall-collapse diagnosis + capped-precision fix, demonstrated by the uncapped ablation* — is diluted to a footnote if the headline mechanism becomes two other papers' methods (LoRA + Nash-MTL) stacked on top. A method section that reads "we use LoRA (Hu et al.), FiLM (Perez et al.), Nash-MTL (Navon et al.), decoupled training (Kang et al.), logit adjustment (Menon et al.), and our Kendall caps" is a systems report, not a contribution.
- **Nash-MTL specifically:** the 204 implementation is not the published algorithm (§2.4); the real one is expensive per step and unstable at effective batch 4; and swapping the gradient-management method *now* would confound the very ablation (capped vs uncapped Kendall, with PCGrad held constant) that is your Figure 1. **Keep Kendall-caps + PCGrad as the method.** Nash-MTL's proper home is one row of the ablation table in a camera-ready with time to spare — implemented correctly, or not at all.
- **What survives from the adapter research:** Phase-1-style warm-starting (train heads/adapters briefly with a frozen backbone before joint training) is cheap, safe, and doesn't have to ship in the final model — adapters can be merged (LoRA is mergeable: `W ← W + (α/r)BA`) or discarded after warm-up. If run13 shows early-epoch instability when the backbone unfreezes, a 2-epoch frozen-backbone warm-up of the heads is the 20-line version of everything Phase 1 promises.

---

## 6. Q4 — Which of the 12 options in 203 are load-bearing?

Ranked by expected-evidence-per-hour, with the gate that unlocks each. The first two items are not in 203 — that is the point.

**The 80% (do these, in this order):**

| # | Item | Cost | Why load-bearing | Gate |
|---|---|---|---|---|
| 0 | **Push run12's real state; run overfit probe; launch ST baselines** | ½ day + GPU-2 | They decide whether *anything* in 203 is needed. Five rounds pending. | None. Do first. |
| 1 | **Detection augmentation** (mosaic/copy-paste; `--det-aug` exists, inactive) | ~0 code, restart | Zero params; largest published upside for 27K-frame detection; attacks the actual constraint (data volume) | Probe says eval works |
| 2 | **6D pose + geodesic loss** (203 §4.1) | 2 h, ~+0.3M | Cheap, correct (Zhou 2019), upgrades your *headline* head — the positive-transfer candidate — and gives the paper a principled pose section. Build `R_gt` from fwd/up (§2.6 caveat) | None (orthogonal to other heads) |
| 3 | **PSR operating point: threshold sweep + transition-aware weighting** (203 §2.2, loss only) | 2 h, 0 params | Directly attacks focal-collapse-to-negatives, the live failure mode behind "loss low, F1 zero" | Probe on PSR |
| 4 | **Activity: fix logit-adjust wiring (loss-side, raw-logit eval) + decoupled classifier retrain** (204 §4) | 2 h config/loss | Below-random is a weighting/prior pathology; these are the two principled long-tail fixes, both ~free | Probe on activity; verify label space |
| 5 | **Kendall ablation run (`--kendall-uncapped` — wire it for real) + log_var logging** | 1 h + 1 run | It IS Figure 1. Also §1b's starvation diagnostic for free | None |

**The conditional 20%:**
- **BiFPN@128 + per-level TAL top-k** — legitimate detection architecture step *after* items 0–1 show TAL is alive but weak. Fix the duplicate-`p5_lat` bug and use GroupNorm (§2.5).
- **Temporal attention pool for activity** — medium risk (requires re-plumbing `MViTFeaturePyramid` to surface per-frame tokens). Gate: ST-activity baseline substantially beats MTL-activity *after* item 4 — i.e., proof the shared cls_token, not the long tail, is the bottleneck. Until then it's speculative surgery on the backbone interface.

**Cosmetic now (skip):** GFLV2 quality branch (+~1 AP on COCO does not transfer to a 24-class head sitting at 0.0 — hunt decimals after you have integers), FiLM, LoRA store, Nash-MTL, detection-conditioned PSR (buggy spec + dead dependency), distillation, VideoMAE, Ego4D. This matches 205's Tier 3/4 — 205's discipline is correct; hold to it.

---

## 7. Q5 — Does the SlowFast-Hydra-MTAN recommendation change the direction?

**No. It is a distraction, and a quantifiably bad trade even on its own terms.**

1. **It starts 4.86 points behind on your best-anchored task.** SlowFast scores 60.39 top-1 on IndustReal activity vs MViTv2-S's 65.25 (your own benchmark table, 198 §2.1). The one task where you have a clean, same-backbone SOTA comparison — the paper's most defensible anchor — gets structurally worse, and the "apples-to-apples with the WACV SOTA" argument dies with it.
2. **MTAN replaces your contribution instead of supporting it.** MTAN's per-task attention masks are a 2019 answer to the same question your Kendall-caps + PCGrad stack answers. Adopting it converts your methodological contribution into "we used MTAN," a reviewer-visible downgrade from *novel diagnosis* to *known technique*.
3. **ASFormer is the wrong regime** (§4, point 4) — it's a full-video segmentation decoder bolted onto a windowed per-frame problem.
4. **47.4M vs 48.6M buys nothing.** Parameter parity, no pretraining advantage (SlowFast's K400 checkpoint is *weaker* than MViTv2-S's), and a full rewrite of a codebase whose current model has never had its mandatory experiments run. The opportunity cost is measured in the same units as Risk 1: weeks of not-running-the-probe.

**Disposition:** one paragraph in the paper's related-work/alternatives-considered, citing the reasons above. The only transferable idea — per-task feature *selection* from a shared trunk — you already implement more cheaply via per-head feature routing (cls_token / P5 / FPN levels), and it's working (pose healthy, PSR responsive to routing).

---

## 8. Q6 — Are we still over-investing in architecture before diagnosis? (Honest verdict)

**Yes — and this round found a worse version of it.** The precise, fair accounting:

**What improved:** 202–205 are better documents than round 3's. They have a priority tiering that puts experiments first, a risk register whose Risk 1 correctly names the pattern, and a final recommendation ("Do NOT start implementing items 5–15 yet") that agrees with me. The PSR diet — if it exists — is exactly what Round 4 ordered. The instinct to consult *before* building the 12-item list was right.

**What got worse:** Round 4's mandatory actions — probe, baselines — are still unrun after being "the next thing" in rounds 1, 2, 3, and 4. The round produced ~1,500 lines of new documentation and **zero pushed lines of code and zero result artifacts.** And a new failure mode appeared: **206 asserts five implementations that the repository does not contain** (§1). Rounds 1–4 deferred experiments; Round 5 is the first to *report work as done* that cannot be verified. That is a category change. The fabricated-efficiency incident (167/170) was caught by an audit and fixed with `efficiency_audit.md`; the immune response exists — but it triggered *after* the fact both times. The 22-agent research compounds the pattern at smaller scale: at least six claims that don't survive contact with a primary source or a calculator (§2.1–2.5), because volume of research was mistaken for validation of research.

**Is 202–205 "preparatory analysis that makes the experiments meaningful"?** Partially — the tiering and specs will save time *when the gates open*. But preparation that repeatedly displaces the thing it prepares for is procrastination with a bibliography. The experiments were meaningful four rounds ago; they need no further preparation.

**The structural fix (adopt as standing rule):**
1. **Push-before-prompt:** no consultation round opens until the repo contains the previous round's code diff and ≥1 result artifact (eval JSON / log) from its mandatory experiments.
2. **One-in-one-out:** no new architecture document until one Tier-1 experiment result is committed.
3. **Claims carry line numbers:** any "implemented/enabled/fixed" in a doc must cite a commit hash. (201 and this file already follow the convention; extend it to your own docs.)

---

## 9. Q7 — The one thing to do tomorrow (plus the EP10 tree, corrected)

**The single highest-leverage action: run the overfit probe on detection.** Half a day, GPU-local, no restart of run12 needed. It resolves the longest-standing unknown in the project (is detection's 0.000 an eval-harness failure or a learning failure?), it gates every detection item in 202–205 (aug, BiFPN, GFLV2), and it is the experiment that has been ordered and deferred five times. If the probe overfits 50–200 clips and eval *still* reads 0.0 → fix the harness and every prior detection conclusion is void; if eval reads high on the overfit set → the harness is fine, and features/data are the real gap → activate `--det-aug` and proceed down §6.

**But the honest "tomorrow" is three parallel actions, none blocking the others:**
1. **Morning (30 min):** commit and push the actual run12 state — diet PSRHead, logit-adjust wiring (fixed per §2.6), `--kendall-uncapped`, the probe script, the run12 launch command and log path. Until this happens, every number 206 reports is unverifiable, including EP10 itself.
2. **Morning (½ day, GPU 1 or spare capacity):** the overfit probe — detection first, then activity, then PSR, using the *committed* script so the artifact is reproducible.
3. **Morning (30 min setup, GPU 2):** launch `train_st.py --task pose` and then `--task act` staggered. They are the denominator of the entire paper, they take days, and they have been launchable since commit `f2b01cc4a`. ST-pose especially: it decides whether your headline is "positive transfer" or "no cost," and pose needs no architecture changes to be final.

**EP10 decision tree — 206's version, corrected:**

| EP10 (when it can be verified) shows… | 206's action | Corrected action |
|---|---|---|
| Detection > 0.01 mAP | "TAL working, proceed with BiFPN" | Proceed with **augmentation first** (zero params), BiFPN second (with §2.5 fixes). Also log `log_var_det` — if pinned near 4.0, fix the cap before crediting/blaming architecture (§1b) |
| Activity > 5% top-1 | "logit-adjust fixed it, proceed with temporal pool" | **Do not proceed to temporal pool.** 5% says the collapse is breaking, not that the cls_token is exhausted. Next lever is decoupled classifier retrain + more epochs; temporal pool waits for the ST-activity gate (§6). Verify the logit-adjust wiring (§2.6) before attributing anything to it |
| PSR F1 > 0.02 | "P5 fix real, proceed with detection-conditioning" | P5 fix real → **threshold sweep + transition-aware weighting first**; detection-conditioning stays gated on det mAP > 0.3 *and* a per-frame classifier (§4) |
| Any head ~0 | "run overfit probe FIRST" | ✅ Agreed — and this branch is live *today* regardless of EP10, because the probe is independent of run12's progress |
| Pose ≤ 9° | (not listed) | Leave it alone; switch heads to 6D+geodesic only at the next planned restart, not as a hot change |

---

## 10. The corrected two-week plan (what survives from 205 §2)

205's day-by-day plan is structurally fine; re-gate it so architecture cannot start before diagnosis finishes, and cut the items rejected above:

- **Days 1–2 (gate G0):** push run12 state; probe all 4 heads; launch ST-pose, ST-act (GPU 2); wire+launch the uncapped-Kendall ablation run when a GPU frees; verify activity label space + unweighted eval. **Exit criterion: probe artifacts committed, ≥2 ST runs training, EP10 numbers in the repo.**
- **Days 3–5 (gate G1 = probe verdicts):** eval fixes if the probe demands them; `--det-aug` restart; PSR threshold sweep + transition-aware loss; logit-adjust wiring fix + decoupled retrain config; 6D pose head ready for the restart. Launch ST-det, ST-psr as GPU 2 frees.
- **Days 6–9 (gate G2 = detection alive per G1):** BiFPN@128 (fixed) + per-level top-k; run13 = diet-PSR + fixed-logit-adjust + 6D-pose + det-aug (+BiFPN if G2 passed) to ep30–50.
- **Days 10–14:** MTL/ST ratios + CIs, E8 gradient heatmap, Kendall-collapse Figure 1 (capped vs uncapped), efficiency table (48.6M-class model vs ~100M specialists, fvcore), paper L2+L3+method.
- **Not in the plan:** VideoMAE, Ego4D, LoRA/FiLM store, Nash-MTL, detection-conditioned PSR, distillation, SlowFast rewrite, temporal attention pool (parking lot behind its ST gate).

The paper spine is unchanged for the fifth consecutive round, because it was right the first time: **Kendall-collapse characterization + fix (methodological contribution, proven by the ablation), measured per-task transfer map (MTL/ST ratios with CIs), and a genuine ~2× parameter efficiency at single-pass latency.** Everything above either feeds one of those three or doesn't belong in this paper.

---

## 11. ONE-SCREEN SUMMARY

```
206 ASSERTS                              →  ROUND-5 VERDICT (repo + primary sources)
─────────────────────────────────────────────────────────────────────────────────
"We did the diet/logit-adjust/flag/      →  NOT IN THE REPO. PSRHead still 70.9M (:297-326,
 probe/ST-fix; run12 live at 48.6M"         :445); no logit_adjust at :436; no --kendall-
                                            uncapped; no overfit_probe.py; last code commit
                                            pre-dates Round 4. Push it or it didn't happen.
"3 levers: pretraining, det neck,        →  1 of 3 survives. VideoMAE case rests on a wrong
 adapters"                                  number (ViT-B ≈81.5 K400, not 87.4). Per-task
                                            backbone LoRA ⇒ FOUR forward passes ⇒ deletes the
                                            single-pass claim (the surviving efficiency half).
                                            Det neck: keep, gated behind probe + augmentation.
"1.78M PSR vs 5.2M det-conditioned?"     →  Trust the diet. The 5.2M spec has a confirmed bug
                                            (mean-pool→expand ⇒ identical logits on all 8
                                            frames ⇒ cannot localize transitions) and depends
                                            on a 0.000-mAP detector. Fix the operating point
                                            (threshold sweep, transition weighting), not capacity.
"Nash-MTL replaces PCGrad"               →  204's pseudocode is not Nash-MTL (Gα=1 ≠ GᵀGα=1/α).
                                            Swapping methods mid-paper deletes Figure 1. Keep
                                            Kendall-caps+PCGrad; Nash-MTL = later ablation row.
"SlowFast-Hydra-MTAN rewrite?"           →  NO. Starts −4.86 top-1 on your best-anchored task,
                                            MTAN replaces your contribution, ASFormer is a
                                            full-video decoder aimed at 8-frame windows.
"Are we over-investing in architecture?" →  YES, round 5 of 5 — plus a new failure mode: work
                                            reported as done that the repo cannot verify.
                                            Standing rule: no round N+1 until round N's code
                                            diff + ≥1 result artifact are pushed.
ALSO FOUND: code caps are det:4.0/pose:4.0 (not 1.5/2.0 as documented) ⇒ detection has the
WEAKEST starvation floor (exp(-4)≈0.018). Log the log_var trajectories — that's Figure-1
evidence or a live bug, for free. Logit-adjust forward applies at eval too (:285-289) ⇒
as wired it self-cancels; move it into the loss, evaluate raw logits.
─────────────────────────────────────────────────────────────────────────────────
TOMORROW (three parallel actions):
  1. Push run12's real code state + launch log (30 min). Unverifiable = nonexistent.
  2. Overfit probe, detection first (½ day). Fifth round it's been ordered. It gates ALL
     of 202-205's detection items and settles eval-bug vs learning-gap.
  3. Launch ST-pose then ST-act on GPU 2 (30 min setup). They are the paper's denominator.
PAPER SPINE (unchanged, round 5): Kendall-collapse fix + measured MTL/ST transfer map
+ genuine ~2× efficiency at single-pass latency. Protect all three; everything else is a menu.
```

*Companion to 181 (round 1), 186 (round 2), 192 (round 3), 201 (round 4). Same authority rule: where this file and 202–206 disagree, this file is grounded in the committed code, hand-checked arithmetic, and primary sources, and supersedes them. The short version: you asked me to verify an architecture strategy, and verification found that the strategy documents contain real errors (a wrong SOTA number underpinning Lever 1, an adapter mechanism that breaks your own latency claim, a PSR spec that can't localize transitions, a Nash-MTL that isn't Nash-MTL) — but the deeper finding is that the repository cannot confirm the round's premise. The diet is right. The neck is right, later. The probe is right, NOW, and has been for five rounds. Push the code, run the probe, start the baselines — and then 202–205 become what they should be: a well-organized menu you order from based on evidence, instead of a fourth consecutive substitute for it.*
