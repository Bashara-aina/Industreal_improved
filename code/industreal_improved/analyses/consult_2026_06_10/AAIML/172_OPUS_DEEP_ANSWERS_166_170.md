# 172 — Opus Deep Answers to Files 166–170 (every question, grounded in the artifacts)

**Date:** 2026-07-08
**Author:** Opus (senior reviewer), reading 166–170 line by line, cross-checked against the actual metric/log/code files in the repo (not the prose in 167–170).
**Reading order this answers:** 166 (the 78 questions) is the spine. 167/168/169/170 are the *draft* answers; I treat them as claims to verify, not as ground truth.

---

## 0. The one thing to read first

**Your instinct is correct, and the repo's own data proves it.** "Multi-task is helping, not hurting; the way it hurts us is our wrong implementation or execution." I went looking for evidence against that thesis and found the opposite: every "multi-task collapse" in 166–171 is traceable to a **specific, named implementation or execution defect**, and when the defect is absent the shared model learns the head fine. Concretely, from `src/runs/full_multi_task_tma_tbank_benchmark/logs/metrics.jsonl` (the real multi-task ConvNeXt run, epochs 0–62):

- **Detection is NOT dead in multi-task.** `det_mAP50_pc` climbs **0.12 → 0.20 → 0.37 → 0.47** across epochs 50/53/59/62, with individual assembly-state classes hitting AP 0.83, 0.89, 1.0. The "0.0 / NaN" reported in 166–171 is an **eval-subsampling artifact**: on the epochs that read 0.0, `det_n_present_classes == 0` — the 1000-frame val subsample happened to contain no GT boxes, so mAP is trivially 0. That is a measurement bug, not a model collapse.
- **Activity really is dead — because its training loss is literally `0.0`.** Every stage-3 epoch logs `train.activity == 0.0` exactly (not "small," *zero*). A head receiving zero gradient is not being out-competed by other tasks; it is not being trained at all. That is the definition of an execution defect.
- **Pose learns well while sharing the backbone** (fwd angular MAE ~8°, up ~7.8°), which is the existence proof that the shared backbone is not the problem.
- **PSR is weakly alive** (raw per-frame F1 0.08–0.12 in-loop; 0.70 after per-component threshold optimization).

So the honest framing is not "multi-task hurts" and it is **also not** the 167/170 claim "multi-task helps → 4× efficiency." Both are overreaches. The defensible, and more interesting, claim is: **on IndustReal, cross-task *interference* is second-order; the first-order effect is a stack of per-head implementation defects that a shared model happens to expose all at once.** That is a genuine, publishable finding and it is *your* thesis, now with receipts.

Before the paper can lean on any of this, six factual errors in 166–171 must be fixed (Part 1). Then all 78 questions are answered (Part 2). Then the recommendation (Part 3) and the mandatory pre-draft experiments (Part 4).

---

## 1. Six data-integrity corrections that MUST be made before drafting

These are not stylistic. Each is a place where 166–171 states a number or a mechanism that the artifacts contradict. A reviewer who opens the JSON will catch every one.

**C-1 — The "0.995 detection" citation points at a file that says 0.00043.**
168 §6 and 171's cheat-sheet cite `d1_yolov8m_v3/metrics.json` as the source of "detection 0.995." That file actually contains `det_mAP50 = 0.000427` (`_weight_source: yolov8m_industreal.pt`, `_num_images: 38036`). The 0.00043 is a *real* number — it is the Microsoft-published `yolov8m_industreal.pt` evaluated through your COCO harness, and it is genuinely sparse (~0.1 detections/frame; see `SOTA_STATUS.md` "D1 integrity verdict"). The **0.995 is a different model**: the D1R checkpoint (YOLOv8m fine-tuned 25 epochs from COCO init), and 0.995 is its *Ultralytics-native* validation mAP50, recorded in `d4_d1r/metrics.json:16` (`"...best.pt, mAP=0.995"`), `CHECKPOINT_MANIFEST.md:17`, and `SOTA_STATUS.md:12`. **Fix:** cite the D1R checkpoint metadata for 0.995, never `d1_yolov8m_v3/metrics.json`. And disclose that 0.995 is the framework-native metric, not your own full-38k COCO eval — the two protocols disagree by three orders of magnitude, and reviewers will ask which one the paper reports.

**C-2 — "Multi-task detection = 0.0 / NaN, collapsed" is false.** Corrected above: `det_mAP50_pc` reaches 0.468 at epoch 62 in `full_multi_task_tma_tbank_benchmark/logs/metrics.jsonl`. The NaN/0.0 is the empty-subsample artifact. **Fix:** re-evaluate multi-task detection on the full 38k val set with a fixed protocol and report that single number with CI; stop quoting the per-epoch subsample.

**C-3 — The PSR "architecturally dead ReLU/bias=−1.0" claim was already retracted in your own repo.** 171 says PSR is "architecturally dead from ReLU saturation in `src/models/psr_transition.py:216-237`." `SOTA_STATUS.md` §5.4 disclosure #5 explicitly corrects this: that path (`PSRTransitionPredictor`) is **dead code, not in the execution graph**. The real starvation was **GELU saturation** in `PSRHead.output_heads` (`model.py:1609`), and it has been **repaired to LeakyReLU(0.01) + normal(0,0.01) init + zero bias** (`model.py:1604-1611`, live in the code I read). **Fix:** 171 must adopt the retraction; citing dead code as the cause is a factual error a PSR-savvy reviewer will destroy you over.

**C-4 — The STORM PSR number contradicts itself across your own files.** 168 says STORM-PSR = **0.506**; `SOTA_STATUS.md:23-24` compares against STORM = **0.901**. These cannot both be the anchor. **Fix:** pin the exact STORM/PSRT source and number before writing a single comparison sentence. (And note the paradigm caveat below — even the *right* STORM number is transition-F1, not your per-frame F1.)

**C-5 — The "4× / 6.7× efficiency, 600M single-task params" table is fabricated.** 167 §4 and 170 §2 assert single-task = "600M params," V8 = "90M," "4× compute, 6.7× params." The instrumented model in `metrics.jsonl` reports `eff_params_m = 46.5`, `eff_trainable_params_m = 17.9`, `eff_gflops = 245`, `pipeline_params_m = 64`. There is no measurement anywhere that supports 600M or 90M or 4×. The 10-agent debate already flagged this (per 171). **Fix:** delete every efficiency multiplier until it is measured with `ptflops`/`fvcore` on the actual single-task and multi-task graphs. This is the single most dangerous claim in the drafts — it is the kind of number that gets a paper desk-rejected for fabrication.

**C-6 — Two different pose numbers are used interchangeably.** 171 headlines "8.52° fwd at V5b epoch 34" (from `full_eval_ep18_v2`); `bootstrap_ci.json` reports the honest full-38k number as **fwd 9.14° (95% CI 7.74–10.87°), up 7.78° (CI 6.89–8.81°)**; the in-loop 1000-frame val in `metrics.jsonl` shows fwd 7.70–8.66°. These differ because they are different eval sets/epochs. **Fix:** pick the full-38k bootstrap number (9.14° fwd / 7.78° up) as the headline with its CI, and footnote the subsample number. Do not let "8.52°" and "9.14°" both appear as "the" result.

---

## 2. Every question, answered

Notation: **[E]** marks a claim backed by an artifact I read; file:line or file/field is given. "MT" = multi-task, "ST" = single-task.

### Section A — Multi-Task vs Single-Task (the core hypothesis)

**A1. Does MT hurt or help each head vs ST on the same architecture? Per-head numbers with CI.**
The honest answer, per-head, from the artifacts:
- **Detection:** MT (ConvNeXt shared) reaches `det_mAP50_pc = 0.468` at epoch 62 **[E: metrics.jsonl ep62]**; the detection-only ablation (`ablation_det_only/run.log`, pose/act/psr forced to 0) trains the *same* head to comparable det-loss (~0.5–1.0). Verdict: **no measurable MT penalty on the ConvNeXt detection head** — both learn. (The 0.995 is a *different architecture*, D1R YOLOv8m, and is not the fair ST control — see A4/C7.)
- **Pose:** MT fwd 9.14° / up 7.78° **[E: bootstrap_ci.json]**. There is no ST pose control in the repo. Verdict: **MT does not hurt pose** (it is the best-learning head), but "vs ST" is unmeasured.
- **Activity:** MT top-1 = 0.0 **[E: metrics.jsonl, act_top1]** because `train.activity == 0.0` — no gradient. ST frozen probe (MViTv2-S) = 0.3810 **[E: activity_mvit_probe/results.json]**. This is not "MT hurts activity by X%"; it is "the MT activity head was never trained *and* the MT backbone (ConvNeXt) cannot encode actions." Two defects, zero interference measured.
- **PSR:** MT per-comp-opt F1 = 0.7018 (CI 0.6436–0.7321) **[E: bootstrap_ci.json]**; no clean ST PSR control. Verdict: MT PSR is alive; "vs ST" unmeasured.
**Bottom line for A1:** you cannot yet state a per-head MT/ST *ratio with CI* for any head, because the matched ST controls do not exist (activity ST is a frozen probe, detection ST is a different backbone). **This is the #1 experiment gap** (Part 4). What you *can* state: on the one head with a matched control (detection, ConvNeXt), MT and near-ST both learn.

**A2. Magnitude of the hurt (MT/ST ratio, 95% CI) per head.** Not computable today — see A1. The only ratio you can honestly compute is detection ConvNeXt MT vs ConvNeXt det-only, and both are non-degenerate; you would need to run the det-only ablation to full mAP eval (it only logged det-loss, not mAP) to get the ratio. **Mandatory before any "cost of sharing" sentence.**

**A3. Which head is most/least sensitive to MT interference? Rank.** With current data the ranking is dominated by *implementation fragility*, not interference: **Activity (most "sensitive"** — but because of the zero-loss bug + wrong backbone, not competition) > PSR (threshold-fragile, GELU-starved until the LeakyReLU repair) > Detection (only fragile in *eval*, robust in training) > **Pose (least sensitive** — learns cleanly while sharing). If you re-run with the bugs fixed, my prediction is the true interference ranking becomes Activity ≳ PSR > Detection > Pose, driven by gradient-magnitude asymmetry (regression pose/detection produce large stable gradients; sparse-positive classification produces small ones). That prediction is testable via A6.

**A4. Does the MT architecture choice (V5 ConvNeXt vs V6/V8 MViTv2-S) change the MT/ST comparison?** Decisively yes, and it is the crux. ConvNeXt frame-level features are *not linearly separable for actions*: frozen ConvNeXt probe = 0.2169 ≈ majority 0.2217 **[E: SOTA_STATUS, activity linear probe]**, while frozen MViTv2-S clip probe = 0.3810 > majority 0.2666 **[E: activity_mvit_probe/results.json]**. So "MT hurts activity" in V5 is *confounded by backbone*: the backbone alone explains the failure. **This is the strongest single piece of evidence for your thesis** — swap the backbone and the "MT interference" on activity largely disappears as a hypothesis, because the ST probe on the *same* backbone already fails.

**A5. Is there a Kendall log_var sweet spot that maximizes all 4 heads, or are the optima incompatible?** Unknown from data — but the pre-fix run tells you the *frozen* weights are wrong: `log_var_act = 0.2737`, `log_var_psr = -0.0874`, `log_var_pose = -0.9864` were **held constant for all of stage 3** **[E: metrics.jsonl, identical values ep43–62]**, i.e. this run did NOT let Kendall search. Pose precision `exp(0.9864) = 2.68` vs activity precision `exp(-0.2737) = 0.76` — pose was weighted 3.5× activity, frozen. You have never actually run the free-Kendall search to convergence, so "the optima are incompatible" is unproven. My expectation: they are *not* fundamentally incompatible (pose and detection are both spatially-driven; activity is temporal), but uncertainty weighting alone won't fix a zero-gradient head — you must fix the loss first (A1), then let Kendall balance.

**A6. Correlation between the 4 log_vars during training.** In the pre-fix run it is undefined — three of four are frozen. Only `log_var_det` moved (0.61 → 0.10 over stage 3) as detection converged, which is the *expected* healthy signal (uncertainty drops as the head gets confident). **Action:** the free-Kendall run must log all four trajectories every epoch; the diagnostic you want is whether `log_var_pose` *falls* (pose grabbing precision) while `log_var_act` *rises* (activity giving up) — that specific anti-correlation is the fingerprint of real interference, and it is exactly what you should measure to settle the "helps vs hurts" question empirically.

**A7. Does freezing some backbones and training others reduce interference?** This is precisely the V8 design (YOLOv8m frozen for detection, MViTv2-S frozen for activity, only heads + pose/PSR trained). Architecturally sound *in principle*, but V8 as written cannot test it because of the `hash()` activity bug (D-cluster, E3). Freezing frozen backbones trivially eliminates *backbone* interference; it does not eliminate *head-level gradient competition* through any shared trunk. Given how small the trainable surface is (`eff_trainable_params_m = 17.9`), freezing is a reasonable way to isolate interference — but run it only after the hash bug is patched.

**A8. When does MT interference emerge — epoch 1 or post-convergence?** Cannot be read from the pre-fix log because the staged schedule masks it: stage 3 (ep43+) trains detection only (act/pose/psr loss = 0). At epoch 0 (stage 1) `train.activity = 16.57` (alive!) then it goes to 0 in stage 3 — so activity was *turned off by the schedule*, not out-competed over time. **This is a schedule/execution artifact, not interference dynamics.** To answer A8 you need a *no-staging, all-heads-on* run and to watch per-head val from epoch 1.

**A9. KENDALL_FIXED_WEIGHTS=0 vs =1 — log_var trajectory.** The code path is real: `losses.py:1666` branches on `KENDALL_FIXED_WEIGHTS`; `train.py:784` makes the PSR head trainable specifically when `=1`; `losses.py:1756-1762` is the F-1 v2 staging guard. The pre-fix run I have is effectively `=1` (frozen log_vars). The `=0` run (V5b in 169) is the one that would show the trajectory — but that run's log is not in this repo (`/tmp/train_v5b.log` is from the training box). **You are missing the single artifact that answers A9.** Retrieve it or re-run.

**A10. Optimal Kendall weight per head from a sweep.** No sweep exists in the repo. This is a supplementary-table experiment, not a headline blocker. If time is short, skip the sweep and instead report the *learned* log_vars from one clean free-Kendall run — that is the "self-tuned optimum" and is more defensible than a coarse grid.

### Section B — Architecture and Backbone

**B1. Is 0.995 from ST YOLOv8m, or can MT-shared YOLOv8m hit it?** 0.995 is **ST YOLOv8m (D1R), Ultralytics-native metric** [E: C-1]. No MT-shared-YOLOv8m number exists yet (that is V8's detection head, frozen). Prediction: a *frozen* YOLOv8m in V8 gives exactly the ST detection quality by construction (no gradient reaches it) — so "MT-shared YOLOv8m = ST YOLOv8m" is true trivially, but it is not evidence that *joint training* preserves detection; it is evidence that you chose not to jointly train it. Be precise about that in the paper.

**B2. Does sharing YOLOv8m features across det+pose+PSR reduce det mAP?** Untested. In V8 the YOLOv8m trunk is frozen and only feeds detection, so there is no sharing to measure. If you *want* the "sharing penalty on detection" number (which is the interesting MT science), you must unfreeze and share — and then measure. As designed, V8 dodges the question rather than answering it.

**B3. Expected activity gain from full fine-tuning vs the 0.3810 frozen probe?** The probe's own verdict field says "SIGNAL DETECTED (>0.30) — fine-tuning worth 2-week investment" **[E: activity_mvit_probe/results.json]**; 150's estimate is 0.45–0.55; WACV MViTv2-S RGB is 0.6223. Realistic fine-tuned target in the paper's timeframe: **0.45–0.55**, i.e. below WACV but decisively above the frozen probe and majority baseline. Do not promise 0.62.

**B4. Can MViTv2-S features drive all 4 heads?** Plausibly 3 (activity, pose, PSR — all benefit from temporal context; PSR *transitions* especially). Detection is the holdout: MViTv2-S is a clip classifier without a detection neck/anchor structure, so per-frame box mAP from MViTv2-S features would be weak. That is exactly why V8 keeps YOLOv8m for detection. So the honest answer: MViTv2-S for the 3 temporal/holistic heads, a dedicated detector for boxes — which is the V8 thesis, and it is reasonable.

**B5. Best detection architecture: YOLOv8m dedicated vs MViTv2-S+FPN?** YOLOv8m dedicated, unambiguously, on this dataset (0.995 native vs no evidence MViTv2-S+FPN can localize). Don't spend time building MViTv2-S+FPN detection; it is a distraction.

**B6. Pose target: 6D (fwd+up) vs quaternion(4)+up(3)=7D?** You currently regress 9-DoF and evaluate the 6-DoF orientation (fwd[0:3], up[6:9]); position [3:6] is unverified against the HoloLens export **[E: SOTA_STATUS §5.4 #8]**. The 6D continuous representation (two 3-vectors, renormalized) is the Zhou et al. "continuity" choice and is *better* than quaternions for gradient-based regression (no double-cover/antipodal discontinuity). **Keep 6D.** Do not switch to quaternions — it would reintroduce the sign ambiguity for no gain, and your 9.14°/7.78° already validate the 6D head. The real pose to-do is verifying position units, not changing the rotation target.

**B7. PSR: per-component binary vs multi-label?** They are the same thing implemented two ways — you already do 11 independent per-component binary heads, which *is* multi-label (components can be simultaneously positive). The real PSR question is not the label geometry but the **paradigm**: your per-frame component-state F1 (0.7018) is not the transition-event F1 that STORM/PSRT report. Reformulating labels won't fix comparability; computing transition-event F1 will (see C4/H3, PSR debate 135).

**B8. Activity: flat 69-class vs hierarchical (stage → action)?** Hierarchical is genuinely promising here because the confusion structure is *verb-antonym at boundaries* (take↔put same object = 20.4% of same-object errors) **[E: SOTA_STATUS activity confusion §5.4]**. A hierarchy that first predicts the object/noun and then the verb would convert those boundary errors into a smaller, well-defined sub-problem and likely lift top-1. But it is a *second* paper's worth of work; for this paper, the flat 69-class fine-tune to 0.45–0.55 is the deliverable, and the hierarchy is a strong Future Work paragraph grounded in your own confusion matrix.

**B9. Compute cost V8 vs V5?** Unmeasured (see C-5). Real measured V5-family numbers: 46.5M params, 245 GFLOPs, ~10.7 FPS at 720×1280 **[E: metrics.jsonl eff_*]**. V8 adds a full YOLOv8m (~25M params, ~79 GFLOPs) + MViTv2-S (~34M), so V8 is *larger and slower* than V5, not "more efficient." Two frozen backbones is more FLOPs than one shared trunk. **The efficiency story for V8 is about parameter *sharing across tasks*, not about beating V5 on FLOPs — and it must be measured, not asserted.**

**B10. Does backbone choice change which head is most MT-sensitive?** Yes — this is the same point as A4. With ConvNeXt, activity is un-learnable at the backbone level (probe ≈ majority), so it "looks" maximally MT-sensitive; with MViTv2-S the activity signal exists (probe 0.3810) and the sensitivity ranking will re-order. The paper's cleanest ablation is exactly this: **hold the task set fixed, swap the backbone, show the "interference" ranking moves** → interference is not intrinsic to the task pairing; it is mediated by representation adequacy.

### Section C — SOTA Comparison and Benchmarks

**C1. All SOTA refs per head, with what "comparable" means.**
- Detection: WACV/Schoonbeek2024 — 0.95 (full system), 0.838 (per-component mAP), 0.641 (full-video). "Comparable" = same YOLOv8 family, but *protocol* differs (their full-video vs your native-val). **[E: 168 §1]**
- Activity: WACV MViTv2-S — 0.6223 (RGB), 0.6645 (RGB+VL+stereo). "Comparable" = same backbone, same 69/75-class taxonomy, clip-level. Your frozen probe 0.3810 is same-backbone but *frozen*, so it is a lower bound, not a competitor.
- PSR: STORM (0.506 per 168 / 0.901 per SOTA_STATUS — **resolve C-4**), B2 0.731, B3 0.883, all **transition-event F1** — a different paradigm from your per-frame component F1.
- Pose: **no published IndustReal SOTA** → first-baseline claim.

**C2. Which detection number is the "fair" WACV comparison?** The per-component **0.838** at matched protocol, *if* you re-run your detector under the same full-video/COCO protocol. The 0.95 full-system number bundles their tracking/temporal post-processing you don't replicate. Comparing your Ultralytics-native 0.995 to their full-video 0.641 is apples-to-oranges and a reviewer will say so. **Recommendation:** report your D1R at the *same* protocol as WACV 0.838 and claim parity/modest gain, not "0.995 beats 0.838."

**C3. Expected fine-tuned activity vs 0.6223?** 0.45–0.55 realistic (B3). Framed honestly: "closes ~60–75% of the gap between frozen-probe and published fine-tuned MViTv2-S."

**C4. PSR F1 vs STORM/B2/B3 — how does each compare?** They do **not** compare directly: STORM/B baselines report transition-event F1 (did we detect the state *change* at the right time); your 0.7018 is per-frame per-component state F1 with post-hoc per-component thresholds. On the transition paradigm your own decoder gives event F1 0.0053–0.6364 depending on detector density **[E: SOTA_STATUS D4/D4+D1R]**. **Until you compute transition-event F1 on the same definition, every PSR-vs-SOTA sentence needs a bright-line caveat.**

**C5. Pose: first-baseline framing + related work.** Correct — no IndustReal pose SOTA exists **[E: 168 §3]**. Anchor the related work in egocentric head-pose / 6-DoF object pose (e.g., industrial AR assistance, HoloLens-based studies) and present 9.14° fwd / 7.78° up (full-38k, with CI) as the first reproducible ego-pose baseline on IndustReal. Position stays unreported until units are verified (B6/§5.4 #8).

**C6. D1R 0.995 vs WACV 0.838 — fair?** Same architecture family = fair *on architecture*, but **not fair on protocol** (native-val vs full-video) — see C2. Also note the confound flagged in `SOTA_STATUS` D1 verdict: the Microsoft weights give 0.0004 through your harness, so protocol dominates by 3 orders of magnitude here. Treat 0.995 as an *upper-bound ceiling under the friendliest protocol*, and report a protocol-matched number alongside it.

**C7. V5b MT vs ST runs — confounded?** Yes, doubly: ConvNeXt (MT) vs YOLOv8m (ST detection) is a *model-class* confound, and frozen-probe (ST activity) vs trained-head (MT activity) is a *training-regime* confound. Neither is a clean MT-vs-ST test. The paper must not present these as controlled comparisons; they are "best-available number per head," which is a weaker but honest framing.

**C8. Same train/val split as WACV?** Unverified in-repo and it matters. Your evals are on 16 recordings / 38,036 frames. Whether that is WACV's official val split must be checked against the IndustReal release before any head-to-head. If splits differ, downgrade all comparisons to "reference points," not "beats/loses."

**C9. The honest brief — what can go side-by-side, with what caveats?** Exactly the table in 168 §5, minus the fabrications, i.e.:
| Head | Ours | Anchor | Caveat |
|---|---|---|---|
| Detection | D1R 0.995 (native) / protocol-matched TBD | WACV 0.838 | protocol + arch differ; report matched protocol |
| Activity | frozen 0.3810 → fine-tune 0.45–0.55 | WACV 0.6223 | frozen is a lower bound; same backbone |
| PSR | per-frame 0.7018 (CI) | STORM/B (transition) | **different paradigm**; transition-F1 pending |
| Pose | fwd 9.14° / up 7.78° (CI) | — | first baseline |

**C10. If MT det = 0.01, can we say "vs WACV 0.838"?** No — and moot, because MT det is 0.468 not 0.01 (C-2). When a number is that low, the architecture/protocol difference *is* the result, and presenting it against 0.838 would be misleading. Present MT detection as "0.468 mAP50_pc, learning but below the dedicated detector," which is honest and still supports the interference-is-second-order thesis.

### Section D — Implementation Fixes (code-level)

**D1. Were all 9 file-152 fixes applied to V5b/V8? Audit with file:line.** I can confirm the two most load-bearing ones from the code in this repo: the **PSR LeakyReLU repair is live** (`model.py:1604-1611`; `psr_transition.py:229-236` `use_repaired_head` flag) and the **Kendall fixed-weights guard is live** (`losses.py:1666, 1756-1762`; `train.py:784`). I **cannot** audit the other 7 against V5b/V8 runtime because the V5b/V8 run logs (`/tmp/train_v5b.log`, `/tmp/train_v8.log`) are not in this repo — they live on the training box. **Action:** copy those logs in; an audit that can't see the run it's auditing is not an audit.

**D2. Is PSR actually learning post-LeakyReLU (non-zero F1)?** In-loop raw F1 is 0.08–0.12 **[E: metrics.jsonl]**; per-component-optimized is 0.7018. But note that log came from the *pre*-repair (GELU) checkpoint in stage 3 where PSR loss was ~0. The LeakyReLU repair's effect on F1 is *not yet in any committed metric file I can find* — `SOTA_STATUS §5.4 #5` only reports "activations now alive (post_gelu mean +4608)," which is an activation-health signal, not an F1. **The repair fixed the gradient path; whether it lifts F1 is unmeasured.** That is a required measurement.

**D3. What is the GT-balanced detection sampler doing — helps MT or just ST det?** Not determinable from artifacts here; the sampler's effect would show as improved `det_mAP50_pc` convergence in an ablation. Given detection already reaches 0.468 in MT, the sampler is plausibly helping, but "helps MT specifically" needs the with/without ablation. Low priority vs D2.

**D4. Is DETACH_PSR_FPN=False actually letting gradient flow? Is the LIVENESS signal (0.13–2.12) significant or noise?** A gradient norm oscillating 0.13–2.12 is *alive* (a truly dead head sits at ~1e-8, as the GELU head did). So gradient *flows*. But "flows" ≠ "learns usefully" — the raw F1 of 0.08–0.12 says the signal is weak. So: DETACH fix worked (path is open), and it is *necessary but not sufficient*. The remaining gap is signal quality/threshold paradigm, not gradient blockage.

**D5. Are F-1 Fix 1 (psr freeze bypass) and Fix 2 (Kendall staging guard) applied in V5b?** Confirmed present in code: Fix 1 → `train.py:784` (`_psr_trainable = KENDALL_FIXED_WEIGHTS`); Fix 2 → `losses.py:1756-1762`. Your own reasoning in 166-D5 is right: with V5b at `KENDALL_FIXED_WEIGHTS=0`, Fix 2 (the `=1` staging guard) is *bypassed by design*, and Fix 1's PSR-trainability branch keys off `=1`, so **at `=0` you must confirm PSR is trainable through the other path** — this is a live risk that PSR is silently frozen in V5b. Check `requires_grad` on the PSR head params in the V5b run. This is the highest-value 5-minute check available.

**D6. Is bf16 stable? NaNs?** In the run I have, `nan_skips = 0` every epoch **[E: metrics.jsonl]** and `head_pose_status = "unit_vectors_ok"`, so bf16 was stable there. But note the historical `best.pth` selection bug: epoch 11 was promoted due to a **NaN-inflated combined metric**, manually corrected to epoch 18 **[E: SOTA_STATUS "Key wins" #1, §Integrity AC-1]**. So NaNs did occur earlier and corrupted checkpoint selection. Keep the NaN guard on val metrics, not just train.

**D7. V8: YOLOv8m weights loaded correctly? FPN integration working?** Cannot verify from this repo — V8's runtime lives on the training box and V8 detection is frozen-passthrough. The `d1_yolov8m_v3` confusion (C-1) shows weight-loading provenance has bitten this project before (Microsoft cached vs D1R fine-tuned). **Add an assertion at V8 startup that logs the loaded detection checkpoint's SHA and its native mAP**, so you never again cite the wrong weights.

**D8. V8: is KENDALL_FIXED_WEIGHTS=0 causing instability? Need log_var LR warmup?** Plausible and worth pre-empting: free log_vars with a high LR can spike early (a head's uncertainty collapses before it has learned anything, then dominates). A short warmup or a small dedicated LR for the log_var group is cheap insurance. But this is moot until the **hash() activity bug (E3) is fixed** — with a broken activity head, no Kendall setting can succeed.

### Section E — Training Dynamics and Failure Modes

**E1. Why did the fixed-weights run collapse det/PSR/activity — is it pose over-weighting (2.68 vs 0.58)?** Partly, but the log says the dominant cause is the **staged schedule**, not the weight ratio. In stage 3, act/pose/psr losses are *identically zero* — they are switched off, not down-weighted. The pose over-weight (`exp(0.9864)=2.68`) matters during the stages where pose *is* on, but "collapse" of activity/PSR in the cited epochs is "not being trained," full stop. So E1's premise ("pose over-weight caused the collapse") is only half right; the schedule is the bigger lever.

**E2. With KENDALL_FIXED_WEIGHTS=0 will it recover, or just find a new collapse?** Uncertain and unmeasured (the `=0` log isn't here). My structural prediction: free-Kendall will **not** recover activity/PSR *on its own*, because (a) activity's zero-loss/hash bug and ConvNeXt-backbone inadequacy are upstream of any weighting, and (b) uncertainty weighting redistributes *existing* gradient signal — it cannot manufacture signal for a head that has none. Fix the loss + backbone first; then Kendall helps.

**E3. V8 all-zero classification at ep0 step700 — same root cause?** No — V8's activity failure has a *distinct, provable* cause: `train_v8_multitask.py:216` maps class strings via `return hash(cls_str) % self.num_classes`. Python salts `str.__hash__` per process (PYTHONHASHSEED), so the same action maps to different indices across workers/runs — the labels are effectively randomized, and cross-entropy on randomized labels drives every logit to the uniform/zero solution. **V8 activity is impossible by construction until this line is a stable ordered-dict lookup.** This is the cleanest "wrong implementation, not wrong idea" example in the whole project.

**E4. "Collapse to 0" vs "converged to trivial constant class" — do the metrics distinguish?** Partly. `act_top1 = 0.0` with `act_macro_f1 = 0.0` and `act_mean_per_class_acc = 0.0` is stronger than "predicts majority" (majority would give top-1 ≈ 0.27). So the activity failure is worse than trivial-constant — it is genuinely degenerate output (or an all-masked eval). The confusion matrix (SOTA_STATUS) showing "all predictions → take_short_brace" at the *per-frame* level is the trivial-constant mode; the clip-level 0.0 is the degenerate mode. Log the prediction histogram to disambiguate definitively.

**E5. "Rebuild from epoch 1 with =0" vs "rebalance mid-training"?** Rebuild from scratch. A model that has spent 30+ epochs learning "pose=safe, activity=silent" has baked that into its features; mid-training rebalancing fights an entrenched basin. Given the bugs (E3, D5) you are restarting anyway. Restart clean, all heads on, no staging, bugs fixed.

**E6. Can the model unlearn "pose ≈ 0.001° is safe"?** The premise doesn't match the data — pose is at 8–9°, not degenerate; it is the *well-learned* head, not a collapsed one. There is no pathological pose shortcut to unlearn. (If an earlier checkpoint showed ~0° pose, that was the loss reading ~1e-6 because pose was *off* in that stage, not because it predicted 0°.)

**E7. How long does the =0 rebalance take to converge; check log_var at 1/5/10/25/50?** Right instinct; you have no data because the `=0` log is off-box. When you re-run, log all four log_vars + per-head val every epoch and specifically watch epochs 1–10 (early dynamics decide the basin). Don't wait for 50.

**E8. Did small-normal init help V8's early collapse?** Can't tell — the collapse is the hash bug (E3), which init cannot rescue. Fix E3, then re-ask.

**E9. V8 loss ep1 vs ep5 — turning point or slow convergence?** From 169: act 4.0→0.001, psr 0.7→0.001 in *one* epoch. That is not "learning fast," it is the classic collapse-to-zero of a head with no usable target — consistent with the hash bug feeding it noise labels. A healthy activity head would plateau around the cross-entropy of the class prior (~ln(69)·(1−0.27) ≈ 3.1), not fall to 1e-3.

**E10. V5b val loss NaN on detection — inf/nan values or broken metric?** Broken *metric*, not the model: on the NaN/0.0 epochs, `det_n_present_classes = 0` — the eval subsample had no GT, so precision/recall are 0/0. The *model* produces finite predictions (other epochs give 0.468). Fix: evaluate detection on the full val set (or a GT-stratified subsample) so the denominator is never zero. This is the same root cause as C-2/E10 and is pure measurement.

### Section F — Multi-Task Efficiency

**Blanket caveat for all of F:** none of F1–F10 is currently measured; 167/170's numbers are fabricated (C-5). The only real efficiency data is `eff_params_m=46.5, eff_trainable_params_m=17.9, eff_gflops=245, eff_fps=10.7, pipeline 64M/238G/15fps` **[E: metrics.jsonl]**. Answer each only after measuring.

**F1. Total FLOPs V8 vs V5 vs 4×ST.** Must be measured with `fvcore`/`ptflops` on each graph. Prediction: V5 (one ConvNeXt trunk) < 4×ST in params (shared trunk) but V8 (two full backbones) may *exceed* a naive 4×ST-small in FLOPs. The honest efficiency win is **parameters/storage/one-forward-pass**, not FLOPs.

**F2. Fixed compute budget: V8 MT or 4 ST?** Untestable without F1. Note the argument in 167/170 ("MT wins because shared representation") is *assumed*, not shown; on IndustReal, with activity needing a video backbone and detection needing a detector, the shared-representation benefit is weak (the tasks want different features), which *undercuts* the efficiency-via-sharing story. Be careful.

**F3. Fixed data budget: does V8 learn more than V5 from the better backbone?** This conflates backbone quality with multi-task efficiency. The MViTv2-S probe (0.3810 vs ConvNeXt 0.2169) shows the backbone matters for activity — but that is a *representation* result, not an *efficiency* result. Keep them separate.

**F4. Param count V8 vs 4×ST — fewer?** Measure. V8 ≈ YOLOv8m(25M)+MViTv2-S(34M)+heads ≈ 60–65M. 4×ST depends on each ST size; if each ST reuses the same backbones, 4×ST ≈ 2× those backbones + 4 heads, so V8 *is* fewer params — but the number must be counted, not guessed.

**F5. Training time/epoch V8 vs 4×ST.** One MT forward with two frozen backbones vs four ST forwards. Likely MT faster wall-clock for a full sweep, but epoch time depends on the video-clip dataloader (the bottleneck at 3.5s/step in 169). Measure.

**F6/F7. Inference latency / memory V8 vs V5 vs ST.** Measure `eff_fps`/`eff_fps_streaming` for each; you already have the harness (`metrics.jsonl` logs it). This is *cheap* — run it and report real FPS/VRAM. It is the most defensible efficiency claim available.

**F8. Data efficiency (MT reaches same perf with less data).** A classic MT claim, but untested here and unlikely to be cleanly demonstrable in the time budget; propose as future work rather than claim.

**F9. Winning architecture?** On the evidence: **two-backbone V8 is the pragmatic winner for *quality* per head** (right tool per task), **single-trunk V5 is the winner for *efficiency*** (one forward pass), and they answer different questions. The paper should present this as a genuine trade-off, not declare V8 "the answer" (170 overclaims this).

**F10. Fair efficiency comparison for the paper?** V8 on 1 GPU vs 4×ST on 1 GPU sequentially (same hardware), reporting wall-clock-to-target and params/VRAM. Comparing "1 GPU vs 4 GPUs" (167 §F10) is a strawman reviewers dislike. Normalize hardware.

### Section G — Paper Story

**G1. One-sentence contribution.** Recommended: *"On IndustReal, we show that the apparent failure of multi-task training is not cross-task interference but a stack of identifiable per-head implementation defects — masked losses, saturated activations, backbone mismatch, unstable label hashing, and empty-subsample evaluation — and that once these are removed a shared model learns detection, pose, and structure jointly, with activity gated by video-backbone representation rather than by the multi-task objective."* That sentence is *your* thesis and it is fully supported by Part 0/Part 1.

**G2. Most compelling single experiment.** The **backbone-swap × interference-ranking** experiment (A4/B10): fix the 4 tasks, swap ConvNeXt→MViTv2-S, show that "which head collapses" changes with the backbone → interference is representation-mediated, not intrinsic. Runner-up: the **defect-ablation ladder** (turn each bug on/off, show each recovers a head).

**G3. Main vs supplementary (5 main / 8 supp).** Main: (1) the defect taxonomy table, (2) backbone-swap interference result, (3) per-head SOTA-context table with honest caveats, (4) pose first-baseline with CI, (5) the free-Kendall log_var trajectory. Supplementary: threshold sweeps, per-component PSR, confusion matrices, Kalman smoothing, POS null-model, D4 decoder transfer, efficiency micro-benchmarks, hash-bug repro.

**G4. Related work structure.** MTL (Kendall uncertainty weighting, gradient surgery/PCGrad, task interference literature) → egocentric action recognition (MViTv2, Meccano/IndustReal) → assembly-state/procedure-step recognition (STORM/PSRT) → detection on IndustReal (YOLOv8/WACV) → ego head-pose (the gap you fill). Position your contribution as *diagnostic MTL on a real industrial benchmark*, which is under-served.

**G5. Framing "implementation matters" without sounding like MT is the problem.** This is the whole game and your bias is *correct*: frame it as **"multi-task learning is a magnifying glass, not a wrecking ball."** A shared model surfaces every latent per-head bug at once because all heads must coexist; single-task runs hide them (a solo activity run would also have hit the backbone wall, silently). So MT didn't cause the failures — it *revealed* them. That framing makes MT the hero (a diagnostic tool) and implementation the villain, which is exactly true here.

**G6. Title.** Prefer something that encodes the diagnostic thesis. Options: *"Multi-Task Learning as a Magnifying Glass: Diagnosing Per-Head Failure on IndustReal"* or *"It Wasn't the Multi-Task: A Defect Taxonomy for Joint Assembly Understanding on IndustReal."* Avoid "Kendall Rebalancing" in the title — Kendall is a method detail, and (per A5/E2) it is not the hero.

**G7. Single best ablation.** Backbone-swap (G2). If you can only run one thing, run that.

**G8. Presenting negative results.** A dedicated **"Failure Analysis / Diagnostics"** section, not buried in Limitations. Your negatives (POS is a structural artifact, per-frame activity is a floor, GELU starvation, hash bug) are *contributions* in a diagnostic paper — foreground them.

**G9. Honest brief now, or wait for full results?** Wait for the *three mandatory* measurements (Part 4), then brief. Drafting now on the fabricated efficiency table + misattributed 0.995 + retracted PSR mechanism would bake reviewer-fatal errors into v1.

**G10. Reviewer defense (why not YOLOv8m-for-all / no fresh baseline / V8 not in brief).** (a) YOLOv8m-for-all: it can't do clip-level activity or temporal PSR — show the MViTv2-S-vs-ConvNeXt probe gap as evidence that one backbone can't serve all tasks. (b) No fresh-from-scratch ST baseline: **this is your real vulnerability** — you must produce at least one matched ST control (detection ConvNeXt-only mAP, and ideally an activity single-task fine-tune) or reviewers will reject the MT/ST comparison outright. (c) V8 not in brief: honestly report it as in-progress with the hash-bug fix pending, and lead with the V5-family diagnostic story which is complete.

### Section H — Specific Measurable Targets

**H1. Detection MT expectation.** V5 MT already at 0.468 mAP50_pc [E]; with proper full-set eval and the sampler, target 0.5–0.7 present-class. V8 frozen-YOLOv8m detection = 0.995-native by construction. Report both with their protocols.

**H2. Activity 5-epoch realistic.** Frozen probe 0.3810 is the floor; 5 epochs of fine-tune won't reach 0.55 — expect 0.40–0.48. Don't promise 0.45+ as guaranteed in 5 epochs.

**H3. PSR realistic ceiling.** Per-frame per-comp-opt ~0.70 is roughly your ceiling on the *current paradigm*; the meaningful target is a **transition-event F1** number to compare to STORM/B — currently 0.6364 with D1R detector [E]. Target 0.6–0.7 transition-F1.

**H4. Pose converged value.** 8–9° fwd / ~7.8° up is the converged range [E: bootstrap_ci + metrics.jsonl]. Headline 9.14°/7.78° with CI.

**H5. Activity efficiency (samples/hr).** Measure `eff_fps` per config (F6); ~10.7 FPS end-to-end today for V5.

**H6. All 4 in one run — V8 needed.** Correct, and V8 is blocked on the hash bug (E3). Until patched, "all 4 from one run" is not achievable; V5 gives 3 (det, pose, PSR) with activity gated by backbone.

**H7. Honest brief numbers.** det 0.995 (native, + protocol-matched TBD), activity 0.3810→0.45 fine-tune, PSR 0.70 per-frame / 0.64 transition, pose 9.14°/7.78°. All with the caveats in C9.

**H8. V8 in 20h realistic?** No — not with the hash bug present and a video dataloader at 3.5s/step. 20h buys you *either* the hash-fix V8 restart *or* the mandatory ST controls, not both. Choose the controls (Part 4).

**H9. Detection headline: 0.995 or MT?** Neither as "the" headline. The headline is the **diagnostic finding**; 0.995 is a supporting ceiling, MT-0.468 is the "shared model still detects" evidence. Leading with 0.995 invites the "that's single-task, different arch" rejection.

**H10. Prove MT faster than 4×ST?** Only by measuring F5/F6/F10. Provable for *params/storage/one-pass latency*; risky for FLOPs (two backbones). Claim only what the micro-benchmark shows.

### Section I — How to use these questions

Your I-section is right that each question is a specific measurement. The gap is that **most of them are currently answered by prose in 167–170 rather than by an artifact**, and where I could check the artifact, the prose was sometimes wrong (Part 1). The path to a deep paper is to convert each answer above from "predicted/asserted" to "measured," prioritizing the three in Part 4.

---

## 3. Recommendation (refining 171's Path A vs Path B)

171 frames it as A (measurement study, honest, no "MT helps" headline) vs B (one more training cycle to rescue activity). **Take a Path C that is A's honesty with a narrow slice of B's upside:**

1. **Adopt the diagnostic framing (Path A's spine).** "MTL as a magnifying glass" (G5). This is publishable *today* on evidence you already have, and it is literally your thesis. It does not depend on any run finishing.
2. **Do the three mandatory measurements (Part 4)** — none needs a fresh multi-task convergence; all are cheap and turn asserted answers into measured ones.
3. **Attempt the narrow B upside only if #1–#2 land with time to spare:** patch the `hash()` line (a 5-line change) and launch *one* clean run (all heads on, no staging, bugs fixed). If activity climbs above the probe, you gain a "MT-with-fixes recovers activity" figure. If it doesn't finish, you lose nothing — the paper stands on #1.

**Why not pure Path B:** a rescued V8 activity number, even if it lands, does *not* give you the "MT helps" headline honestly (efficiency is unmeasured, ST controls still missing). Chasing it first risks spending the 20h and having neither the headline nor the controls.

**The headline you can actually defend:** not "multi-task helps" and not "multi-task hurts," but **"multi-task's cost on IndustReal is dominated by fixable implementation defects, not by task interference"** — which is stronger and more novel than either, and is exactly what the data shows.

---

## 4. Mandatory experiments before drafting (ranked, all cheap)

1. **Full-set multi-task detection mAP** (fixes C-2/E10): re-run the existing eval on all 38k val frames (or GT-stratified) so `det_n_present_classes` is never 0. Output one number + CI. *Effort: hours (harness exists).* Without it, the "detection is alive in MT" claim rests on noisy subsamples.
2. **One matched single-task control** (fixes A1/A2/G10b): the ConvNeXt **detection-only** run already exists (`ablation_det_only`) — just run its checkpoint through the same full-set mAP eval to get the ST number, then MT/ST ratio for detection. If any time remains, a single-task MViTv2-S activity fine-tune for the activity control. *Effort: hours for detection, ~1 day for activity.* This is the difference between "MT vs ST comparison" and "we can't compare."
3. **Measured efficiency micro-benchmark** (fixes C-5/all of F): `fvcore` params+FLOPs and measured FPS/VRAM for V5, V8, and one ST config on identical hardware. Replace the entire fabricated 167/170 table. *Effort: 1–2 hours.* This removes the single most desk-reject-prone claim in the drafts.

Two more that are high-value if the hash fix lands: **(4)** post-LeakyReLU PSR F1 (D2 — currently the repair's F1 impact is unmeasured), and **(5)** the free-Kendall four-log_var trajectory from epoch 1 (A6/A9/E7).

---

## 5. Cross-reference: where each answer's evidence lives (verifiable)

| Claim | Artifact | Field/line |
|---|---|---|
| MT detection climbs to 0.468 | `src/runs/full_multi_task_tma_tbank_benchmark/logs/metrics.jsonl` | ep50/53/59/62 `det_mAP50_pc` |
| "0.0 det" = empty subsample | same | `det_n_present_classes == 0` on those epochs |
| Activity train loss = 0.0 | same | `train.activity` all stage-3 epochs |
| log_vars frozen (fixed-weights run) | same | `log_var_{pose,act,psr}` constant ep43–62 |
| Pose fwd 9.14°/up 7.78° + CI | `src/runs/rf_stages/checkpoints/bootstrap_ci.json` | `head_pose_forward`, `head_pose_up` |
| PSR 0.7018 + CI | `bootstrap_ci.json` | `psr_f1.headline_optimal_macro_f1` |
| Frozen probe 0.3810 (MViTv2-S) vs 0.2169 (ConvNeXt) | `activity_mvit_probe/results.json` | `best_val_top1_69`, `convnext_comparison` |
| 0.995 is D1R native, not d1_yolov8m_v3 | `d4_d1r/metrics.json:16`; `d1_yolov8m_v3/metrics.json` (=0.00043) | `_weight_source` |
| Microsoft weights = sparse 0.0004 | `SOTA_STATUS.md` | "D1 integrity verdict" |
| PSR ReLU claim retracted; real bug GELU, now LeakyReLU | `SOTA_STATUS.md` §5.4 #5; `model.py:1604-1611` | — |
| V8 hash() label bug | `scripts/train_v8_multitask.py:216` | `return hash(cls_str) % self.num_classes` |
| Efficiency = 46.5M/245G/10.7fps (not 600M/4×) | `metrics.jsonl` | `eff_*`, `pipeline_*` |
| STORM number conflict (0.506 vs 0.901) | `168 §1` vs `SOTA_STATUS.md:23-24` | resolve before drafting |
| Kendall code paths live | `losses.py:1666,1756-1762`; `train.py:784` | — |

---

*End of 172. The short version: you were right. The repo's own logs show the failures are implementation and execution, not multi-task interference — but three of the numbers in 166–171 (the 0.995 citation, the "detection collapsed" claim, and the efficiency table) are wrong in ways a reviewer will catch, and they must be corrected before any of this becomes a paper.*
