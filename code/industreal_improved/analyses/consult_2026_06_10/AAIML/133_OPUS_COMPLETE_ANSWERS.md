# 133 — Opus Complete Answers: All 66 Questions, All 30 Debates, Both Supplementary SOTA Files

**Date:** 2026-07-06
**Responds to:** files 127, 128, 129, 130, 131 (AAIML/) **plus** the two supplementary files `127-50-deep-questions-for-opus-sota-comparison.md` and `128-AGENT-DEBATES-sota-comparison.md` (consult root).
**Companion:** `132_OPUS_ANSWERS.md` holds the extended treatment of the top-10 questions, the in-repo verification audit, and the re-sequenced Week-1 plan. This file completes coverage: every question ID gets a verdict, every debate gets a ruling, and the supplementary files are reconciled against the main set.
**Scope reminder:** verification is against the git repo only. Result artifacts on the workstation (SOTA_STATUS.md, all metrics.json, threshold sweeps, train logs, `/tmp/*.log`) remain unverifiable from here — commit them (132 §7).

---

## §0. Cross-Document Contradiction Registry (found while answering — resolve these FIRST)

These are conflicts **between your own documents** that will produce inconsistent paper text if not resolved. None of them is in files 127-131's own issue lists.

**C-1. Who owns mAP50=0.995?** The supplementary files attribute 0.995 to *"our multi-task ConvNeXt-Tiny detection head at epoch 18"* (127-sota Q1 context; 128-sota Debate 1 claim line). The main AAIML files (127/129/130/131) and the .tex attribute 0.995 to the **separately-trained D1-R YOLOv8m** and give the multi-task ConvNeXt 0.358. The repo evidence (`runs/detect/.../d1r/results.csv` — an ultralytics training log with box/cls/dfl losses) proves 0.99484 belongs to the **YOLOv8m training run**. Verdict: the supplementary files carry a stale/incorrect attribution from an earlier SOTA_STATUS revision. Mark both supplementary files SUPERSEDED in their headers. Any sentence of the form "our detection head achieves 0.995" must never reach the paper.

**C-2. Which YOLOv8m did D1 evaluate?** 129 §13.1 says D1 loaded `yolov8m_industreal.pt` (authors' weights). The .tex (line ~187) says *"COCO-pretrained YOLOv8m (no IndustReal training) achieves 0.0 mAP50"*. Debate 1.1's resolution also says COCO. These can't both describe the same run — and the repo explains how the confusion arose: **`eval_yolov8m.py` tries the Microsoft IndustReal weights URL and silently falls back to COCO-pretrained `yolov8m.pt` if the download fails** (docstring + `_download_weights`, lines ~54-102). If the fallback fired, mAP=0.0004 is *expected by construction* (80 COCO classes vs 24 ASD codes) and D1 says nothing about protocol mismatch. Actions: (a) check the D1 v1/v3 logs for the "Downloading IndustReal weights" vs fallback line; (b) make the script **fail hard** instead of silently falling back; (c) rerun D1 once with the verified authors' weights; (d) run the debate-7.1 class-histogram (COCO model predicts spread over 80 IDs; ASD model predicts within 0-23 — instantly distinguishable). Until this is done, every D1 conclusion in 127/129/130/131 is provisional.

**C-3. Activity taxonomy: 47 vs 69 groups.** The .tex reports activity on **47 hybrid verb-grouped classes** ("reduced from 75", line ~96; macro-F1 0.205/0.129 table rows) and Pathology 2 uses "46/74 classes <1%". The current eval stack uses **75→69 hybrid grouping** (repo-verified: `act_remap_75_to_69.json`, num_groups=69, 75 ids; `t3_full_eval.json` protocol) and current numbers are top1 0.023/0.028. The paper draft and the eval stack are on **different label spaces with different metrics from different eras**. Unify on 75→69, re-derive every activity number in the .tex, and fix the 74-vs-75 raw-class count.

**C-4. PSR story: pre-fix vs post-fix.** The .tex abstract/results still tell the *pre-MonotonicDecoder-fix* story (F1=0 from all-ones collapse, 87.1% all-ones frames, POS=0.9693/0.999). The current story (post variable-shadow fix + threshold sweep) is per-frame macro-F1: 10k-subset global 0.10=0.7217, per-comp optimal=0.7499; corrected full-38k values: global 0.10=0.6788, per-comp optimal=0.7018. D4 backbone-swap F1=0 is a separate finding. **The paper's PSR section needs a full rewrite, not an edit** — the headline failure it discloses no longer describes the system.

**C-5. Head pose numbers span three eras.** 7.83°/9.94° (.tex), 8.14-9.14° + up 7.06-7.48° (supplementary), 8.39° + up 26.20°/13.52° (current 129/131). Nothing to debate — this is AC-6 materialized. The freeze protocol (132 §2 Q10) plus a single machine-readable results file that the .tex table macros consume is the fix.

**C-6. Params/GFLOPs disagree.** .tex: 46.47M / 245.3 GFLOPs / 11.02 FPS ("measured"). 129 §5: ~53M / ~93 GFLOPs (citing day1 checkpoint file). A reviewer comparing the architecture table to the efficiency table will catch a 2.6× GFLOPs discrepancy. Re-measure once on the freeze checkpoint with the measurement script committed.

**C-7. Zero-GT class count.** 127 D-4 says "six of 24 classes have zero GT"; debate 1.2 says det_n_present=15/24 (nine absent). The arithmetic supports 15 present: 0.573 × 15/24 = 0.358 — which reveals that the headline 0.358 **is the present-class mAP diluted by zero-GT classes counted as 0**. See D-4 below; this may raise the protocol-correct detection number to 0.573.

---

## §1. Detection (D-1 … D-7)

**D-1 (NaN full eval → what's the true number?).** Extended answer in 132 §2 Q6. Verdict: no publishable full-set number exists today. Separate the failures: D1 = official-weights cross-eval (see C-2), D3 = our model's full-set eval (NaN). For D3: run in-process eval (`EVAL_MAX_BATCHES=0`) — the subprocess path has had 5 bugs and is not worth a 6th debugging round before the deadline; if in-process also NaNs, the bug is in the streaming metric accumulation, not the subprocess, which localizes it. Regardless of outcome, run the 10-seed subsample variance so the paper can print 0.358 ± σ. Priority: Week 1 (unchanged from 130 P1.3).

**D-2 (claim 0.995 beats SOTA?).** Extended answer in 132 §2 Q5. No. 0.995 is the single-task ceiling measurement (repo-verified). Structure: WACV 0.838 is a soft baseline (1 GPU-day beats it on a recording-aware split — itself a finding); multi-task model achieves X% of the ceiling; the measured cost is the contribution. Fix the internally inconsistent phrasing: 0.358/0.995 = **36% of ceiling = 64% cost** (131 says both "64-68% of ceiling" and "-64% cost"; the .tex says "64-68% of the capability" — pick "X% of ceiling" and recompute after D-4 resolves which mAP convention is correct).

**D-3 (IoU/protocol comparability with WACV 0.838).** Both numbers are mAP50 (.tex line 64 confirms WACV's is mAP50 on 24-class ASD), so the 0.358-vs-0.838 gap is not an IoU-protocol artifact. The *real* protocol gaps are: (a) split — WACV's own annotated-frames (0.838) vs entire-video (0.641) numbers show a 0.2 split sensitivity, and D1-R's 0.995 on our recording-aware split proves cross-split comparison is meaningless; (b) zero-GT class handling (D-4). Action: one paragraph in the paper pinning each number to (model, split, frame-set, mAP convention); never put cross-split numbers in the same table column. mAP50-95 as secondary metric throughout (d1r has it: 0.861).

**D-4 (class imbalance; standard vs present-class mAP).** This is more consequential than 127 rates it. Repo-verified: `evaluate.py` computes both `det_mAP50` (mean over all 24 classes, zero-GT classes contributing 0) and `det_mAP50_pc` (present-classes-only, lines ~1645-1684, added as a fix on 2026-06-04). COCO convention **excludes** classes with no GT from the mean — so if WACV followed COCO convention, the protocol-comparable number for our model is **0.573, not 0.358**, and the multi-task result is 58% of ceiling (42% cost), not 36%. The 0.358 = 0.573 × 15/24 identity (C-7) confirms the dilution mechanism. Actions: (1) verify WACV's convention from their eval code/paper; (2) resolve the 6-vs-9 zero-GT count discrepancy; (3) headline whichever convention matches WACV, report the other in a footnote. This is a 30-minute check that could shrink the paper's biggest weakness by a third — highest ratio of impact to effort of anything in file 130.

**D-5 (efficiency thesis vs free YOLOv8m).** The efficiency claim survives only at system level, never per-task: 4 tasks, one forward pass, 46.5M params vs a 4-model pipeline (66.4M + fusion + 4× inference; .tex line 254 already makes this argument). Two exposures to fix: (a) FPS — ours 11.02 on RTX 3060 vs YOLOv8m's published 178 on V100; do not let an efficiency table invite that comparison without hardware normalization; (b) the "sub-$450 GPU" claim — keep the corrected $429 (contribution audit corrected the "$299" error; supplementary Q8). And per supplementary Q8's missing evidence: **Ablation A (single-task same-backbone baselines) is required before "67% fewer params" style claims** — the equal-gradient-update ablation in the .tex partially covers this but is not the same experiment; state clearly which is being cited.

**D-6 (distill 0.995 teacher into ConvNeXt head?).** Endorse as the one *forward-looking* detection experiment (P2.1), with two constraints: time-box it (3 days; if soft-label plumbing fights the RetinaNet head, stop and write it as future work), and do not gate any paper claim on it. Expected honest outcome: +0.1-0.2 mAP, turning the cost section into cost-plus-mitigation. If it works even partially it also answers D-5's "why not just use YOLOv8m" — the answer becomes "we do, as a teacher."

**D-7 (error-state class evaluation).** With 0 GT error-state instances in val, mAP is undefined for class 24 — and that's reportable: run the *false-positive-rate* eval (how often the model hallucinates error_state on error-free frames; cheap counting over existing predictions). WACV's error-state FPR=65% gives a published anchor for exactly this metric. If our FPR is low, it's a genuine differentiated claim ("does not hallucinate error states"); if high, it goes in §5.4. Also confirm from the dataset whether error-state absence in val is by split design — if all error recordings landed in train, say so, since reviewers will ask why the class exists but is never evaluated.

---

## §2. PSR (PSR-1 … PSR-7)

**PSR-1 (POS paradox).** Extended answer in 132 §2 Q3. POS leaves the headline; D4 is the accidental null model proving the artifact; print the null-POS table (all-zeros + copy-previous-frame) in §5.2.1. Additionally (from supplementary Q5): a **POS@tolerance** variant — ordering scored only within ±k-frame windows around GT transitions — is the only version of POS with informational content; 1 day on cached logits; if implemented, report it as the honest ordering metric and relegate raw POS to the appendix.

**PSR-2 (transition F1 vs per-frame F1).** Both, always, same predictions, clearly labeled — this is P2.6 and it is promoted to Week 1 (132 §4) because it decides the PSR narrative (see AC-3). Use 128-sota Debate 2's recommended two-level wording nearly verbatim in the paper; it is the best pre-written paragraph in the entire document set.

**PSR-3 (Kendall kills PSR?).** Extended answer in 132 §2 Q1. No — the evidence points at dead per-component heads (ReLU gating + bias −1.0 saturation; repo-verified architecture), not the 4-8% Kendall down-weight. Order: 1-hour activation diagnostic → head repair (bias 0.0, LeakyReLU/GELU) → warm-start retrain → `KENDALL_FIXED_WEIGHTS=1` as the *ablation* (env var, no code edit — 129 §4's "False" table entry and 130's "toggle in config.py" are both wrong about the mechanism).

**PSR-4 (D4 F1=0 meaning).** Extended answer in 132 §2 Q2. The YOLOv8m detector fired on <1% of frames under Q48 thresholds tuned for ConvNeXt statistics — re-tune the hysteresis on YOLOv8m outputs before the disclosure text is written; the result decides whether §5.4 says "decoder does not transfer" or "decoder transfer requires recalibration."

**PSR-5 (threshold sweep = data snooping?).** The sweep is 19 thresholds × 11 *independent* components = 209 fits, not 19^11 (debate 7.2-B overstates); per-component calibration on 38k frames is standard practice and the +0.028 over global-threshold is plausible as real. But two of your own numbers already bound the fragility: full-set 0.7018 vs 5k-subset 0.7810 — a 0.03 swing across subsets is the same order as the claimed improvement. So: LOO-CV (P2.5) is mandatory before the calibrated number is headline; until then **0.7217 (global threshold) is the honest primary** with 0.7018 as "calibrated (val-selected)".

**PSR-6 (all 11 sub-heads dead → what did the head learn?).** Follows PSR-3's answer, with one sharpening: if the transition heads are dead, the 0.7018 comes from thresholding whatever near-constant logits they emit — i.e., prevalence calibration plus backbone signal leaking through the shared trunk. The decisive cheap analysis: add two null columns to the per-component table — always-positive F1 = 2p/(1+p) per prevalence p (comp 4: 0.249 vs achieved 0.346; comp 10: 0.310 vs 0.402) — the achieved-minus-null delta (~+0.10 on low-prevalence comps) is the honest "learned signal" quantification, and it is nonzero, which rescues the head from the "pure prior" accusation (adversarial debate 10.3) *without* any retraining. Then the head-repair retrain shows how much more was available.

**PSR-7 (sequence-mode only path that trains PSR).** Correct reading: per-frame batches contribute psr=0.0000, so the head sees gradient only on 25% of steps (SEQ_EVERY_N=4), *and* those gradients die at the per-component heads (PSR-6). After head repair, if PSR learns, consider a short PSR-focused fine-tune phase with `PSR_SEQ_EVERY_N_BATCHES=1` (env-overridable; config.py:1671 documents the option) rather than raising the ratio for the whole run. The per-component "appears in sequence batches" analysis (127's missing evidence) is superseded by the head-repair experiment — don't spend time on it.

---

## §3. Activity (ACT-1 … ACT-7)

**ACT-1 (0.028 vs 0.622).** Extended answer in 132 §2 Q4. Architectural ceiling *claim* requires the linear-probe result; probe first (1 day), then decide on TCN+ViT/MViTv2-S. Print the majority-class baseline beside 0.028 — the number is statistically indistinguishable from the prior and the paper must say so before a reviewer does.

**ACT-2 (is a per-frame MLP defensible as the method?).** Only as a *floor baseline* within the multi-task probe framing, paired with the latency argument (per-frame labels at zero marginal latency vs 178-1149 ms clip windows — debate 3.3-B). It is not defensible as "the activity method" of a 4-task system paper. Minimum for review survival: the baselines table (majority-class, linear probe, MLP) + one temporal ablation (TCN+ViT, P1.4) whose outcome is reported either way. If the probe says the backbone is the bottleneck, the honest §5 sentence is "activity requires either a video backbone or temporal pretraining; our per-frame formulation establishes the floor" — and P5.1 (MViTv2-S head) is cut.

**ACT-3 (75→69 verb grouping).** Repo-verified: hybrid mapping, 75 ids → 69 groups. Defensible *if*: (a) both 75- and 69-class numbers appear (supplementary table row per debate 3.2); (b) the three object-similarity merges (pull_small_screw_pin→pull_wheel, tighten/loosen_tooth_washer→nut) are each justified in one sentence or moved to a pure-verb mapping; (c) T3's top1_75 = top1_69 = 0.6223 is cited as evidence the grouping doesn't inflate baselines. **Blocking issue: the .tex is on 47 groups (C-3) — the taxonomy unification precedes any activity text.**

**ACT-4 (per-frame can't separate take/put — provable?).** Yes, and cheaply: compute the confusion matrix of existing per-frame predictions and show errors concentrate within verb-antonym pairs on the same object (take_X↔put_X). That single figure converts an assertion ("temporally ambiguous by construction") into evidence, and it costs zero GPU-hours — the predictions are already cached. Add to Week 1 writing support.

**ACT-5 (which eval protocol?).** Primary: per-frame macro-F1 (class-imbalance-honest) + per-frame top-1. Clip-level 16-frame majority-vote appears only once, labeled as the bridge to the T3 protocol-verification, never as "our clip-level performance". Justify in one line: assembly errors must be flagged at the frame where they occur (the latency argument).

**ACT-6 (train the TCN+ViT head as ablation?).** Yes = P1.4, gated on the probe (ACT-1). Note config.py:960's own comment — the temporal head needs a fresh run because its TCN+ViT weights are randomly initialized; budget the full 2-3 days, don't warm-start expectations.

**ACT-7 (first per-frame baseline claim legit?).** Legitimate with three conditions: the baselines table (so "first" has content), a documented literature search (so "first" is checkable — the industreal-sota-benchmarks survey is the citation), and the design-choice framing *decided explicitly* (debate 3.3 resolution: deliberate-design tone vs limitation-disclosure tone produce different §1s — recommend deliberate-design in §1 + full limitation disclosure in §5.4; they are compatible if the abstract doesn't oversell).

---

## §4. Head Pose (HP-1 … HP-6)

**HP-1 (is "~15° SOTA" real?).** No. Supplementary Q7 traces it to unverifiable search snippets; the benchmark-reference explicitly failed to find the Ohkawa paper on CVF. Hard rule: an uncitable number is a nonexistent number. Remove every "near SOTA"/"~15°" instance (including 129 §1's table cell and SOTA_STATUS) and claim **first ego-pose baseline**, full stop. This makes the 8.39° *stronger*, not weaker — nothing to be measured against means nothing to lose against.

**HP-2 (7.06/13.5/26.20).** Extended answer in 132 §2 Q7. 26.20° full-eval is the number of record pending P2.4's median+IQR per-recording breakdown; 7.06° is pre-fix-era (AC-1-tainted); 13.5° is an unbounded 300-frame subset. Fix 130's success-metric wording (a breakdown can't move MAE to ≤15°).

**HP-3 (position units).** Repo-verified: config.py:853 "UNIT UNCERTAIN — DO NOT REPORT mm/cm", evaluate.py:1969 "do not use for reporting" (both cited in 127/129 with a stale line number — update to :1969). Papers: zero position claims, orientation-only (6 of 9 DoF). P3.3 (author/SDK verification) is post-submission work. Debate 9.2-3's point stands: if half the 9-DoF output is unreliable, say "orientation baseline," not "9-DoF baseline."

**HP-4 (OpenFace/6DRepNet category error).** Proactive distinction wins (debate 4.3 resolution endorsed): one taxonomy paragraph (face-pose: face crop → orientation in face frame; ego-pose: full egocentric frame → wearer head orientation in world frame), cite 2-3 IMU/HoloLens ego-pose works, and **zero** face-pose numbers anywhere. Silence loses because reviewers fill the vacuum with the wrong comparison.

**HP-5 (FiLM ablation missing for pose).** Two distinct gaps are being conflated: the .tex reports a FiLM ablation *on activity* (18.3 vs 16.1, p=0.032 — single seed, fails Bonferroni; debate 9.3-2), and no FiLM ablation exists *for the pose head* at all. Given FiLM is claimed as technical novelty, run the pose-head FiLM-off arm (P3.2, 2 days) **or** demote FiLM to "supporting component" in the contributions list (see A-2). If keeping the activity FiLM ablation, re-run at 3 seeds or soften to "suggestive"; p=0.032 at 1 seed with 4 conditions will not survive a statistics-literate reviewer.

**HP-6 (temporal smoothing headroom).** The repo already contains `src/evaluation/eval_pose_kalman.py` — the experiment 127 lists as "missing" is scaffolded. Run it; report single-frame (deployment-honest) and smoothed (headroom) side by side. If smoothing gives 1-2°, the claim becomes "8.39° single-frame, 6-7° with standard filtering" — strictly better paper, half a day of work.

---

## §5. Architecture (A-1 … A-7)

**A-1 (what does Kendall buy?).** Extended answer in 132 §2 Q8. As deployed — with HP_PREC_CAP, fixed-lambda override, staged-training disabled (config.py:104), and a full env-var bypass — it is bounded-Kendall, not automatic balancing. Present it as such: "uncertainty weighting requires per-task bounds under extreme label sparsity" is the defensible, novel-ish claim; the fixed-weight ablation is its empirical leg. Debate 5.2-B's counterpoint is worth keeping: log_var_det *does* track convergence, so the mechanism isn't dead — it's supervised.

**A-2 (FiLM: modulating or processing noise?).** Repo-verified: pseudo-keypoints from detection argmax, confidence detached at model.py:696, both FiLM inputs detached — no gradient can improve the conditioning signal. So FiLM's benefit, if any, is one-directional modulation from a heuristic signal. Two cheap moves before the expensive ablation: compute L2 of (γ−1) and β across val batches (one forward pass — if γ≈1, β≈0, FiLM is a pass-through and the novelty claim dies quietly); and check whether detection's dead epochs (loss=0.0000 in train.log per debate 5.1-B) make the pseudo-keypoints degenerate. Paper positioning: FiLM is a *component*, the pathology analysis is the contribution — 04_BEST_PAPER_FORMULA's 90/100 novelty score for FiLM should not survive this audit.

**A-3 (sequence-mode 25% overhead for a detached, zero-F1 head).** Decision tree, gated on PSR head repair (PSR-3): if repaired PSR learns → the overhead was justified, report it as a cost line-item; if PSR stays dead → drop sequence mode (P3.5, env/config change) and say so in the fixes catalog. Reviewer B's mitigations are real (sequence batches are smaller; inference-time cost is zero via the causal cache) — include them in the cost accounting rather than the headline 25%.

**A-4 (3-layer transformer on T=1 for 75% of batches).** True and mildly embarrassing but low-stakes: on T=1 the causal transformer degenerates to a per-token MLP with attention overhead. Post-head-repair, either route per-frame batches around the transformer or leave it and report the timing cost in the appendix. Not paper-critical; do not spend deadline time here beyond one timing measurement.

**A-5 (FeatureBank bypassed).** Repo-verified: with ACTIVITY_HEAD_SIMPLE=True the bank stores detached features that the simple head never consumes — it is dead code in the evaluated configuration. Consequence: remove FeatureBank/TMA from the architecture figure and contribution text for *this* paper (they belong to the `full_multi_task_tma_tbank_benchmark` experiment line), or explicitly mark "present but disabled in reported configuration". Claiming disabled machinery is the kind of figure-vs-config mismatch artifact reviewers grep for.

**A-6 (ACTIVITY_GRAD_BLEND_RATIO 0.05→1.0).** Repo-verified: now 1.00 (config.py:981) — i.e., the blend mechanism is fully open = disabled as a *mechanism*. Disclose in the fixes catalog exactly as the skeptic would put it: the gradient-scaling strategy was progressively abandoned; final config uses direct gradients. It's one honest sentence and it inoculates against the "you patched over a broken gradient path" reading (which is otherwise correct).

**A-7 (3.1M PSR head justification).** Deferred to the head-repair outcome: if F1 recovers meaningfully, the capacity is retroactively justified; if not, the honest fix for *this* paper is reporting the head as over-provisioned (a parameter-sensitivity note), not re-architecting before the deadline. The .tex's 4-model-pipeline comparison already books PSR at 3.1M consistently — no contradiction there.

---

## §6. Training Infrastructure (TI-1 … TI-6)

**TI-1 (CUDA crash root cause).** Three live hypotheses (TDR timeout under memory pressure / Blackwell+CUDA13+torch2.12 driver issue / grad-accum backward spike). Ruling: do **not** bisect batch 3/4/5 on the deadline-critical GPU (132 already deprioritized 6.1). The single worthwhile config test is debate 6.1-C's batch_size=4 + GRAD_ACCUM_STEPS=4 (same effective 16, half the accumulation window, ~2× throughput if stable) — run it once for 12h on a *non-critical* run. Also do the zero-cost diagnostics: dmesg Xid codes + temperature logs from the crash windows, which distinguish driver fault from thermal without any GPU time.

**TI-2 (throughput tradeoff bs=6 vs bs=2).** The right metric is epochs/day *including* restart amortization, under which bs=2 already wins (2.2 vs 2.4 samples/s nominal, minus 1-3h crash cycles). No paper claim rides on this; one sentence in the setup + supplementary crash log (debate 6.3 resolution). MTTC measurement is over-engineering — skip.

**TI-3 (PSR zero loss / GELU saturation).** Same root as PSR-3/PSR-6 (note both activation stories — GELU-after-linear64 and ReLU-in-transition-heads — point to activation death at different depths; the diagnostic forward pass in 132 §2 Q1 covers both). Do not "replace GELU with ReLU and retrain" blind (127's proposed fix) — measure first, repair second.

**TI-4 (grad accum × TDR).** Covered by TI-1's bs=4/ga=4 test — it is the same experiment. If stable, it simultaneously answers TI-1, TI-2 and TI-4 and doubles training throughput for P1.1/P1.4. That's why it's the only infra experiment worth pre-deadline time.

**TI-5 (GPU allocation / DDP).** Ruling: no DDP. Heterogeneous DDP (Blackwell 16GB + Ampere 12GB) synchronizes at the slower card, and NCCL on a mixed pair under CUDA 13 is new-territory risk during deadline weeks. The current split (5060 Ti trains, 3060 evals/ablates) is the correct allocation; the §4 Week-1 schedule in 132 keeps the 3060 saturated with decisive experiments, which was the actual inefficiency.

**TI-6 (why crashes began at epoch 12).** Consistent with cumulative causes (fragmentation, thermals, the 3060 ablation starting mid-training as a power/PCIe confound). Cheap mitigation that needs no diagnosis: scheduled clean restarts from checkpoint every ~8-10h of training (the resume path is already battle-tested via crash_recovery.pth). Document; not paper content beyond one supplementary line.

---

## §7. Eval Pipeline (EP-1 … EP-5)

**EP-1 (0-vs-1 class index).** Repo-verified as resolved: eval asserts 0-indexed, "No shift needed" comment; v2's +1-shift arm scored 0.0 (worse), which is the empirical confirmation. The brute-force histogram (debate 7.1 resolution) is still worth its 10 minutes because it *also* resolves C-2 (COCO vs authors' weights produce unmistakably different class distributions). Then EP-1 closes permanently.

**EP-2 (RGB vs BGR eval-vs-training).** Fixed and repo-verified at three sites. Remaining nit from 127: whether the *training* pipeline was consistent — but the fixed scripts only feed YOLOv8 models (BGR-native), which never saw our RGB training path, so the concern doesn't apply to D1/D4; our own model's eval doesn't channel-swap. Add the 5-line unit test (one known frame → assert identical boxes to ultralytics CLI) and close.

**EP-3 (no save-interval in YOLOv8 evals).** Endorse the resolution verbatim: `--save-every N` flag, default 0, docstring note, copying the `eval_activity_clip.py` pattern (save_interval=5000). 30-minute change; do it the next time either script is touched (D4 re-tune, PSR-4).

**EP-4 (threshold selection snooping).** = PSR-5. LOO-CV mandatory before the calibrated 0.7018 (38k) is primary; 38k global-threshold 0.6788 is the honest default meanwhile (10k values 0.7217/0.7499 superseded).

**EP-5 (does 16-frame majority-vote measure anything?).** It measures per-frame accuracy smoothed by mode-pooling — nothing temporal. The proposed shuffled-frame control is vacuous (majority vote is permutation-invariant by construction; the test cannot fail) — skip it, 127's instinct here was wrong. Correct handling is protocol-labeling, per SOTA-4/8.3: the vote appears only as the T3 protocol bridge.

---

## §8. SOTA Comparison (SOTA-1 … SOTA-8, integrating supplementary Q1-Q8)

The supplementary files' Q1-Q8 map 1:1 onto SOTA-1..8; answered jointly. Overriding note: both supplementary files predate the d1r-attribution correction (C-1) and must be marked superseded.

**SOTA-1 / suppl-Q1 (detection claim + D1 audit).** Of the four offered explanations, the evidence now supports a *fifth* the question didn't list: the eval script's silent COCO fallback (C-2) — which is option (a)/(b) by mechanism but with a known, checkable cause. The class-mapping audit (option a) is done (EP-1). Definitive experiment: rerun D1 with verified authors' weights + class histogram. Until then: no SOTA language; 128-sota Debate 1's recommended wording ("…on our validation protocol, exceeding the published result on the same dataset" + cross-eval footnote) is the ceiling of what's claimable, and even that should wait for the weights check.

**SOTA-2 / suppl-Q2 (STORM paradigm gap).** Confirmed: different quantities under the same symbol. Never in one table column. Our transition-F1 (P2.6, Week 1) is the only number that may sit near STORM/B3's, clearly labeled derived-metric. The supplementary file's "approaching STORM-PSR" framing is exactly the trap — the answer to its question is yes, it would be misleading even with close raw numbers.

**SOTA-3 / suppl-Q3 (B3 procedural-knowledge confound).** With the single-task ceiling established (d1r), the "weaker backbone" excuse is retired — the question's premise ("our detection now exceeds 0.838") conflates the ConvNeXt head with d1r (C-1), but the conclusion stands via the ceiling argument. Remaining gap decomposition: paradigm (per-frame vs transition) + decoder (no learned params, no procedural training signal) + thresholds (D4 re-tune pending). The frozen-detection decoder-retrain the file asks for is P5.3-adjacent — stretch, post-freeze. Comparison sentence: "B3-without-procedural-knowledge on real-only data does not exist as a published number; we therefore do not compare" — that's the honest dead end, state it.

**SOTA-4 / suppl-Q4 (T3 match).** Protocol verification only, methods section, using 128-sota Debate 3's recommended wording. The supplementary question's sharpest point — the "match" assumes remapping preserves relative performance, unverified — is answered by T3's own top1_75 = top1_69 = 0.6223 (remapping was performance-neutral *for the baseline*), which is the sentence to include when the verification is described. Never "matching SOTA."

**SOTA-5 / suppl-Q5 (report POS at all?).** Appendix at most, with the null-model table; POS@tolerance is the salvageable variant (PSR-1). The .tex abstract currently leads with POS=0.9693 vs STORM 0.812 — remove during the C-4 rewrite; debate 9.2's "pair POS and F1 everywhere" is the floor, removal from abstract/conclusion is the recommendation.

**SOTA-6 / suppl-Q6 (activity table?).** Clean break, no MViTv2-S row anywhere in results (only in the related-work sentence establishing the paradigm difference). The temptation the file describes ("0.622 vs 0.6525 looks good in a table") is precisely the engineered-comparison reviewers reject; the contribution audit already closed this — reopening it risks the audit's other conclusions being re-examined.

**SOTA-7 / suppl-Q7 (ego-pose positioning).** First baseline, zero implied comparisons, drop "~15°" (HP-1). The file's own evidence (benchmark-reference: source not found on CVF) makes this the only defensible position. Orientation-only per HP-3.

**SOTA-8 / suppl-Q8 (what does the paper lead with?).** The systems/measurement narrative, definitively: "one model, four tasks, measured multi-task cost, three training pathologies, first baselines where none existed, 8 numbered disclosures." Not a SOTA-contender paper — after C-1, exactly zero of the four heads has a defensible beats-SOTA claim, and the two "first baseline" claims + the cost measurement + the pathology analysis are a coherent, reviewable contribution set. Efficiency claims conditional on Ablation A / equal-gradient-update reconciliation (D-5). This also answers the reviewer-matching concern the file raises: submit to the systems/applied track framing, where the pilot study is an asset rather than padding.

---

## §9. Paper Writing (PW-1 … PW-7)

**PW-1 (naming).** Rename uniformly to "per-frame action classification" — and per C-3, the rename and the 47→69 taxonomy unification are one job; doing the rename without the retaxonomy would produce a consistent name on inconsistent numbers.

**PW-2 (presenting 0.028).** Floor-baseline framing with the majority-class row printed beside it, plus the latency motivation (ACT-2/ACT-7). Never "different paradigm" as a standalone shield — that phrasing without baselines reads as evasion (debate 3.3-A's critique is correct about the failure mode).

**PW-3 (claim-strength rubric).** Adopt this rubric and apply it mechanically: **"beats SOTA"** = same metric, same split, cross-evaluated, artifact-verifiable — currently nothing qualifies; **"competitive"** = within 10% relative under identical protocol — currently nothing qualifies; **"first baseline"** = no published prior after documented search — ego-pose orientation, per-frame action classification, per-frame PSR component-state qualify; **"measured cost"** = ratio against self-established ceiling — detection qualifies; **"not comparable"** = paradigm difference, stated once with reasons — activity-vs-MViTv2, PSR-vs-STORM/B3 qualify. Every results sentence gets one of these five labels; anything unlabeled gets cut.

**PW-4 (ablation table design).** One combined table in the main paper (reviewer attention is the scarce resource), per-head expansions in supplementary. Rows gated by what actually completes before freeze: equal-gradient-update, KENDALL_FIXED_WEIGHTS, FiLM (with 3-seed or "suggestive" label), TCN+ViT-vs-MLP, PSR head-repair. An ablation table with "planned" rows is worse than a smaller complete one.

**PW-5 (narrative arc / deployment space).** Pathology-first narrative; pilot stays in the main paper at 0.5 pages (it is the only positive outcome an ML reviewer intuitively trusts, and practical-impact is the venue's axis) but as *supporting evidence*, not a contribution bullet. On debate 9.1-2 (failure hook vs discovery hook): the discovery framing wins — "we characterize three pathologies standard monitoring misses" — because every subsequent number then reads as evidence rather than as confession; align the intro to the abstract (the .tex abstract already uses discovery framing; the outline's Paragraph-1 failure hook is the one to drop).

**PW-6 (where does F1=0 go?).** §5.2.1 adjacent to the PSR results table, standalone subsection, after the Q2/PSR-4 re-tune decides what the sentence actually is. Post-C-4 rewrite, note the disclosure changes character: the system's own PSR is no longer F1=0 — D4's F1=0 is a *transfer* result, which is materially easier to disclose. The abstract keeps one honest PSR clause either way.

**PW-7 (honesty matrix).** Yes — 0.5-page §5.4 with the 8 numbered disclosures enumerated in 132 §5 (numbers attached, each pointing to its table/figure). It only works if every disclosure has a number; adjectives without numbers signal weakness, numbers signal control. Pathology-2-is-theoretical and the AC-1 checkpoint invalidation live in §4/§6 as integrity notes, not in §5.4.

---

## §10. Adversarial (AC-1 … AC-6)

**AC-1 (best-checkpoint was broken).** Freeze protocol per 132 §2 Q10: named+hashed reporting checkpoint, every eval re-run once against it, every .tex number traceable to a run ID. The epoch-11/18 audit becomes a delta-table byproduct. C-5/C-3/C-4 show the contamination is already three eras deep in the draft — this is the highest-leverage *writing-integrity* item in the entire plan.

**AC-2 (0.028 activity in a "five-task" paper).** Resolved by the ACT chain: probe → TCN+ViT if viable → otherwise the paper is explicitly "four tasks + a per-frame probe head," which is honest and reviewable. The adversarial reviewer's sharpest thrust — "you can't claim the head is both too simple to work and a meaningful interference probe" — is answerable but only *with the probe result in hand*: if the linear probe fails too, the head's failure is uninformative about interference and the probe claim must be dropped; if the probe partially works, the MLP-vs-probe delta is the interference measurement. Either way the confirmatory experiment exists and is cheap; run it (this was debate 10.2's demand, and it is legitimate).

**AC-3 (which PSR number is real).** Extended answer in 132 §2 Q9: both, different pipelines and metrics; the missing 2×2 cell (our pipeline, transition metric) is P2.6, Week 1. The "same evaluation protocol" phrasing in 131 §8.2 should be corrected in place — it is the kind of internal overstatement AC-3 exists to catch.

**AC-4 (did PSR train at all?).** Answered by PSR-3/PSR-6: partially — nonzero deltas over prevalence-null on low-prevalence components prove *some* learning; dead per-component gradients prove it stopped early. The single-task PSR ablation the defending author promised in debate 10.3 is the right camera-ready commitment; pre-deadline, the null-delta table + head-repair delta are sufficient evidence.

**AC-5 (Kendall spiral never observed).** Retract-and-reframe now, no reproduction attempt pre-deadline (132 §3 ruling). Verify during the C-4/.tex pass that Pathology 2's text says "theoretical analysis with preemptive guarding" — the .tex currently presents the fixed-point math (fine) but the framing must not claim empirical observation anywhere, including the abstract's pathology list.

**AC-6 (all numbers will change).** Correct, and now *demonstrated* by this audit: the .tex, the supplementary files, and the current eval stack are three different result-eras (C-1..C-6). Freeze protocol + single machine-readable results source feeding the .tex tables (a `results_frozen.json` the LaTeX macros read) is the permanent fix; adopt at Week-4 freeze.

---

## §11. Rulings on All 30 Debates (37 resolution rows of 128 + 3 supplementary)

Format: **Endorse** (resolution correct as written) / **Amend** (correct direction, changed substance) / **Reject** (resolution wrong or superseded). Amendments reference the section above that argues them.

| Debate | Resolution in 128 | Ruling |
|---|---|---|
| 1.1 detection SOTA | 64-68% ratio, drop BEATS SOTA | **Amend**: ratio arithmetic inconsistent (D-2); ratio may change under present-class convention (D-4); resolution's "COCO classes don't map" explanation of D1 now hinges on C-2 weights check |
| 1.2 NaN full eval | fix subprocess / in-process; 10-seed variance | **Endorse** — in-process first, 10-seed non-optional (D-1) |
| 1.3 cost vs competitive | cost framing, efficiency table | **Endorse** (D-5 adds the FPS-exposure caveat) |
| 2.1 POS paradox | disclose artifact | **Amend**: D4 already *is* the null demonstration; add explicit null table + POS@tolerance (PSR-1) |
| 2.2 backbone swap | controlled ablations | **Amend**: threshold re-tune on YOLOv8m stats precedes any conclusion (PSR-4) |
| 2.3 Kendall suppression | architectural fix (probe, activation test) | **Endorse the debate's own resolution**; note 130 P1.1 contradicted it — restore (PSR-3) |
| 3.1 MLP vs MViTv2 | probe → TCN+ViT → majority baseline | **Endorse + enforce the order**; probe gates all temporal-head spend (ACT-1) |
| 3.2 verb grouping | both class counts, audit merges | **Endorse**; blocked by C-3 taxonomy unification (ACT-3) |
| 3.3 per-frame reframing | latency argument | **Endorse with condition**: baselines table mandatory or the reframing reads as retreat (ACT-7) |
| 4.1 forward MAE normalization | verify checkpoint + normalization | **Endorse** — fold into freeze re-run (AC-6); angular cosine metric is scale-invariant so risk is low |
| 4.2 up-vector | median + IQR, full-eval number | **Endorse** (HP-2); fix 130's ≤15° pseudo-target |
| 4.3 OpenFace strategy | proactive taxonomy, drop face-pose | **Endorse** (HP-4) |
| 5.1 FiLM novelty | show modulation magnitude | **Amend**: γ/β stats first (1 forward pass), ablation second, demote novelty claim regardless (A-2) |
| 5.2 Kendall auto-balance | fixed-weight ablation + log_var figure | **Endorse**; present as bounded-Kendall (A-1) |
| 5.3 sequence-mode overhead | drop if F1=0 at epoch 100 | **Amend**: gate on head-repair outcome, not epoch 100 — the deadline arrives first (A-3) |
| 6.1 CUDA crash cause | bisect batch 3/4/5 | **Reject bisect pre-deadline**; single bs=4/ga=4 test + free dmesg/thermal forensics (TI-1) |
| 6.2 effective batch 16 | accept with caveat, 10-epoch comparison | **Amend**: accept + document; skip the 10-epoch comparison unless bs=4/ga=4 is stable and makes it free (TI-2) |
| 6.3 GPU allocation para | 1 discussion para + supplementary | **Endorse** — supplementary-weighted version (TI-5/TI-6) |
| 7.1 class mapping | brute-force histogram | **Endorse** — now doubles as the C-2 weights discriminator (EP-1) |
| 7.2 threshold overfitting | LOO-CV | **Endorse**; 0.7217 primary until it passes (PSR-5) |
| 7.3 crash recovery | --save-every flag | **Endorse** (EP-3) |
| 8.1 detection SOTA | conditional on D1 audit | **Amend**: audit incomplete until C-2 weights identity is resolved (SOTA-1) |
| 8.2 PSR comparison | separate metric, paradigm section | **Endorse**; use 128-sota Debate 2 wording (SOTA-2) |
| 8.3 activity claim | T3 = protocol verification | **Endorse**; use 128-sota Debate 3 wording (SOTA-4) |
| 9.1 naming | grep + rename | **Amend**: rename + taxonomy unification are one job (PW-1, C-3) |
| 9.2 0.028 framing | different paradigm + latency | **Amend**: paradigm-alone insufficient; majority-class row mandatory (PW-2) |
| 9.3 claim language | rubric before writing | **Endorse**; rubric supplied (PW-3) |
| 9.4 ablation design | run both, pick | **Amend**: one combined table, only completed rows (PW-4) |
| 9.5 narrative arc | deployment 0.5-1.0 pages | **Endorse**; discovery hook over failure hook (PW-5) |
| 9.6 F1=0 disclosure | standalone, paired with results | **Endorse**; content pending D4 re-tune; character changes post-C-4 (PW-6) |
| 9.7 honesty matrix | try 0.5 page | **Endorse**; only with numbers attached (PW-7) |
| 10.1 best-checkpoint | audit + re-derive | **Endorse → freeze protocol** (AC-1) |
| 10.2 activity 0.028 | re-frame as probe / retrain | **Amend**: probe-result-conditional — the interference-probe claim is only valid if the linear probe partially succeeds (AC-2) |
| 10.3 POS inconsistency | disclose D4 vs ours | **Endorse** + fill the 2×2 with P2.6 (AC-3) |
| 10.4 Kendall collapse | fixed-weight ablation | **Amend**: head repair first, fixed-weights as ablation arm (AC-4) |
| 10.5 Kendall spiral | reproduce or retract | **Retract-and-reframe now**; no reproduction spend (AC-5) |
| 10.6 numbers will change | rerun all evals | **Endorse → freeze + results_frozen.json** (AC-6) |
| suppl-D1 detection | internal achievement + footnote | **Amend**: claim ceiling blocked on C-2; supplementary file itself superseded on attribution (C-1) |
| suppl-D2 PSR | two-level reporting wording | **Endorse verbatim** — best pre-written paragraph in the doc set (SOTA-2) |
| suppl-D3 activity | protocol-verification wording | **Endorse verbatim** (SOTA-4) |

---

## §12. Supplementary Files — Disposition

Both `127-...-sota-comparison.md` and `128-AGENT-DEBATES-sota-comparison.md` are **superseded** on the central factual point (C-1: 0.995 attributed to the ConvNeXt head) and on head-pose numbers (8.14-9.14°/7.06-7.48° era). Their *reasoning* survives — the recommended wordings in 128-sota Debates 2 and 3 are adopted verbatim (§8), and 127-sota's Q5 (POS@tolerance) and Q8 (Ablation-A gate on efficiency claims) contributed actions the main files missed. Add a superseded-banner line to both files pointing at 131/133 so future sessions don't re-ingest the stale attribution.

---

## §13. Consolidated New Actions (beyond 132 §4's Week-1 plan)

Everything in 132 §4 stands. This audit adds, in priority order:

1. **C-2 D1 weights identity check** — read D1 logs for the fallback line; make `eval_yolov8m.py` fail hard on download failure; rerun D1 with verified authors' weights + class histogram. (0.5 day, RTX 3060; blocks all D1 conclusions.)
2. **D-4 mAP convention check** — confirm WACV's zero-GT handling; if COCO-convention, the multi-task detection number becomes 0.573 (58% of ceiling) and every cost sentence changes. (30 min + one email/code-read.)
3. **C-3/C-4 .tex reconciliation pass** — taxonomy 47→69, PSR section rewrite, abstract POS removal, pose-number refresh; do after the freeze so it's done once. (1-2 days writing.)
4. **ACT-4 confusion-matrix figure** (verb-antonym pairs; zero GPU). **HP-6 Kalman run** (`eval_pose_kalman.py` exists; 0.5 day). **PSR-6 null-delta columns** (analysis only). **D-7 error-state FPR count** (analysis only).
5. **TI bs=4/ga=4 single stability test** — only on a non-critical run; if stable, all subsequent training halves in wall-clock. (12h passive.)
6. **Superseded banners** on both supplementary files; stale-reference fixes (evaluate.py:1969, KENDALL_FIXED_WEIGHTS env mechanism) in 127/129/130.

---

## §14. Bottom Line

All 66 questions now have verdicts, all 30 debates have rulings, and the supplementary files are reconciled. The audit surfaced seven cross-document contradictions (§0) that none of the source files list — the two that materially change the paper are C-2 (D1 may have silently evaluated COCO weights, voiding the current D1 narrative) and D-4/C-7 (the 0.358 headline is a zero-GT-diluted number; the protocol-correct figure may be 0.573, shrinking the paper's biggest weakness by a third). The strategic conclusions of 132 are unchanged and strengthened: repair the PSR head before believing any weighting ablation, run the four cheap decisive experiments before the expensive ones, freeze results once, and write the paper the rubric in PW-3 permits — a measurement-and-pathology paper with two first-baselines, one honestly-measured cost, and eight numbered disclosures.

**End of 133. Read with 132 (top-10 depth + verification audit + Week-1 schedule).**
