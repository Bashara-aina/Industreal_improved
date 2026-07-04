# 118 — Opus Answers to the Complete 111–117 Document Set

**Generated:** 2026-07-04
**Scope:** This single document answers every open question, anomaly, and decision point raised across the seven-document consultation package:

| Doc | Title | What this document answers from it |
|-----|-------|-----------------------------------|
| 111 | Overview v2 | All 25 open questions in Section 7 (7.1–7.25) |
| 112 | Training Metrics Deep Dive | All 7 anomalies (Section 12) + trajectory interpretation |
| 113 | All-Fixes Chronicle | Untested-fix risk triage (Section 1.3) — what must be verified before publication |
| 114 | Comparability vs 4 Papers | Final comparability rulings per metric, per paper |
| 115 | Execution Plan to SOTA | Revised GPU allocation, calendar, and go/no-go gates |
| 116 | Winning AAIML Synthesis | Verdicts on the 7 contributions, fallback narrative, headline table strategy |
| 117 | 50 Deep Questions for SOTA | All 50 questions answered with verdict, expected value, and sequencing |

**Reading protocol:** Section 0 is the decision summary — if you execute only Section 0, you capture ~80% of the value in this document. Sections 1–7 give the full reasoning, one section per source document. Section 8 is the consolidated priority queue.

---

# Section 0: Executive Verdict and the Ten Decisions That Matter

## 0.1 Overall state assessment

The project is in the strongest state of its history. All five heads are gradient-alive (LIVENESS_GRAD step 1001, epoch 12), the Kendall dynamics are healthy and moving in the correct direction (lv_act descending 0.505 → 0.381 across epoch 12 — activity *gaining* gradient share as it becomes confident, per 112 Appendix D), detection broke its historical ~0.21 ceiling (0.208 → 0.317 between epochs 8 and 11), and the OneCycleLR schedule is only now at peak, meaning most learning capacity is still ahead. The 3060 is idle while the highest-value experiments in the entire plan (D1/D3/D4 plus four inference-only wins from doc 117) cost under a GPU-day combined.

The central strategic error to avoid this week is spending the 3060 on T2 (temporal activity, 3–4 days) before running the cheap experiments that determine whether T2 is even worth doing.

## 0.2 The ten decisions

1. **Run D1 → D3 → D4 on the 3060 today** (5–6h total). These unlock detection and PSR-F1 comparability — the two claims most likely to draw desk rejection if absent. Nothing else on any track has comparable value-per-hour. (115 Track B; 117 Q41/Q16/Q40.)
2. **Run the four zero-training experiments against the epoch-11 checkpoint in parallel:** TTA (Q50), Soft-NMS (Q1), PSR per-component thresholds (Q18), canonical-order POS baseline (Q43). Combined cost ~1–2 days of engineer time, near-zero GPU. Plausible combined effect: +0.03–0.07 detection mAP, PSR F1 0.144 → ~0.17–0.22, and de-risking of the paper's flagship POS claim.
3. **Do NOT start T2 until T3 (MViTv2 remap) completes.** T3 is the 1-day experiment that tells you where the remapped SOTA bar actually is. If remapped MViTv2 macro-F1 lands ≥0.25 (Q45's own hypothesis says 0.25–0.35), the temporal head's expected ~0.15 is 43–60% of SOTA — a weak result bought with 3–4 GPU-days. In that case skip T2 and adopt the per-frame-baseline framing (116 Contribution 4).
4. **After D-experiments, the 3060 runs A2–A4 single-task ablations, not T2.** The paper's core thesis is efficiency; that claim is currently *unsupported* because the only ablation (A1) came out inverted (0.184 vs 0.317) and is triple-confounded. Efficiency evidence > activity evidence, given activity is the acknowledged weakest contribution.
5. **Do not interrupt the main run for an OHEM ablation yet.** Set a decision gate at epoch ~30: if mAP50_pc has not crossed ~0.55 OR cls_mean continues drifting below −9 while mAP plateaus, launch Q5 (OHEM-off + gamma_neg=2.0) as a 25-epoch side run on the 3060.
6. **Freeze the body-pose branch now (stop-grad); remove it in every fresh run.** Do not surgically remove it mid-run — that invalidates checkpoint resume. (111 §7.1.)
7. **Fix the two bookkeeping bugs before anything is published:** (a) `det_n_present_classes=0` in all RF4 validations (112 Anomaly 2) — it touches best-model selection; (b) the ablation checkpoint-directory misrouting into `full_multi_task_tma_tbank/` (111 §3.4).
8. **Verify F22/F22b on the real GPU eval path before POS=0.968 appears in any abstract.** Those fixes were verified on CPU synthetic data only (113 §1.3). D3 doubles as this verification — one more reason to run it today.
9. **Dual-track venue strategy: submit the ego-pose + per-frame + PSR-POS short paper to ICHCIIS-26 (July 15), full paper to AAIML 2027.** Ego-pose is the only zero-caveat contribution; get it time-stamped. Word the short paper as preliminary results.
10. **Publish 6-DoF orientation only; drop position (mm) from all claims.** Report both mAP@0.5 (0.317, headline for comparability) and mAP50_pc (0.506, honest companion) in one table with n_present=15/24 disclosed. Full-disclosure caveat block (~10 lines) covering: POS paradigm, split difference pending D1, per-frame vs temporal, $299 promotional vs $429 MSRP.

---

# Section 1: Answers to Document 111, Section 7 (The 25 Open Questions)

## 7.1 Body-pose dead code: remove, freeze, or leave?

**Answer: Freeze now (option 2), remove in all fresh runs (option 1 deferred).**

Reasoning: The body-pose branch (Wing Loss on pseudo-keypoints from detection boxes) contributes a stable-but-meaningless loss of ~0.80–1.11 that *drives the shared pose Kendall weight*, while the real task (head_pose, loss 0.023 at epoch 11) is buried under it. HP_PREC_CAP already neutralizes the head-pose-dominance failure mode, but the weight allocation remains distorted. Freezing (requires_grad=False on the body-pose sub-head, zero its loss term) costs nothing, is checkpoint-compatible, and removes the distortion. Surgical removal mid-run would change the state-dict shape and break `--resume` against epoch-11 checkpoints — not worth it with 88 epochs remaining. In every fresh training run (T2, A2–A4, any backbone swap), delete the branch outright: 1.6M params freed, Kendall cleaner, one fewer reviewer question. Document in the paper as a limitation of the current run ("a vestigial keypoint branch shares the pose uncertainty parameter; we verified its loss is near-constant and its gradient contribution negligible").

## 7.2 (first) Detection ceiling: structural or fixable?

**Answer: Let the run continue; gate at epoch ~30. The evidence favors continued improvement, with one warning signal to watch.**

For continued improvement: (a) mAP50 rose 52% in 3 epochs (0.208 → 0.317) and the LR schedule only peaked at epoch 10–11 — the model has trained at peak LR for roughly one epoch; (b) anchor quality is healthy (527 positives/image, mean IoU 0.879, max 0.999); (c) DET_PROBE verdicts are LOCALIZING with max IoU 0.942; (d) ConvNeXt-Tiny already exceeded the ResNet-50 ceiling (0.317 vs 0.207), so the old plateau was at least partly backbone-limited, not purely OHEM-limited.

The warning signal: cls_mean drifting from −6.87 to −8.87. That is the classifier moving toward "everything is background" — the exact signature the OHEM+FocalLoss suppression theory predicts. It has not yet hurt mAP, but it is the leading indicator to watch.

Concrete gate: at epoch 30, if mAP50_pc < 0.55 or cls_mean < −9.5 with a flat 3-epoch mAP trend, launch Q5 (OHEM disabled, gamma_neg=2.0, 25 epochs on the 3060) as the definitive test. Do not stop the main run for it; the two can run concurrently once A2–A4 complete.

## 7.2 (second) Activity: invest in temporal or accept per-frame?

**Answer: Gate on T3. Run T3 first (1 day); only run T2 if the remapped MViTv2 bar comes back ≤ ~0.20.**

The 5–6-day T2 investment buys an expected macro-F1 of ~0.15. Whether 0.15 is presentable depends entirely on the remapped MViTv2 number, which is unknown until T3 runs — and Q45's own hypothesis estimates 0.25–0.35, in which case 0.15 reads as "less than half of SOTA," strictly worse for the paper than the clean "first per-frame baseline" framing. If T3 lands ≤0.20, T2's 0.15 becomes "75% of SOTA without Kinetics pretraining or multi-modal input," which is publishable. This is a textbook case of spending 1 day to price a 5-day bet before placing it.

Independent of that decision, three cheap activity interventions from doc 117 (Q9 blend ratio, Q10 GT-fraction, Q47 FeatureBank) can move the per-frame number itself — see Section 7 below.

## 7.3 PSR F1 gap: can D4 bridge 0.144 → ~0.60?

**Answer: The 0.50–0.70 estimate is plausible but unproven; run Q17 (tau distribution) and Q18 (thresholds) alongside D4 to decompose the gap correctly.**

The F1@±3 metric has a hard timing tolerance. If the MonotonicDecoder's median detection delay exceeds 3 frames (Q17's hypothesis: 3–5 frames typical, >5 for rare components), then F1 is *structurally capped by the paradigm* regardless of how good the ASD inputs are — and D4 with YOLOv8m inputs would come in below expectations for reasons that have nothing to do with detection quality. Sequencing: run Q17 first (hours, inference-only on existing predictions). If tau is inside tolerance for most components, D4's 0.50–0.70 range is credible and detection quality is confirmed as the bottleneck. If tau is outside tolerance, the paper's PSR story changes from "detection-limited" to "paradigm-limited, fixable with per-component thresholds (Q18) and temporal smoothing (Q19)" — still an honest and publishable narrative, but a different one. Set expectations at 0.45–0.65, with the D4 result plus Q17 jointly determining which narrative is true.

## 7.4 Ego-pose: publish now or hold for AAIML?

**Answer: Publish now at ICHCIIS-26; it strengthens rather than weakens AAIML 2027.**

The ego-pose baseline is the project's only zero-caveat contribution: no comparison target, no paradigm disclosure, no metric dilution. Risk of holding: someone else notices IndustReal ships HoloLens 2 head tracking and publishes first — the barrier to entry is low for anyone with the dataset. Risk of publishing: essentially none if the ICHCIIS paper is explicitly framed as preliminary; the AAIML paper then cites it as "previously established baseline, extended here with the full multi-task system," which is a *stronger* position (external validation) than claiming everything at once. Standard practice; conferences do not treat a 4–6-page HCI short paper as novelty-destroying prior art for an 8–12-page ML paper with four additional contributions.

## 7.5 3060 crash diagnosis

**Answer: Memory pressure at batch_size=6 is the primary suspect; the fix stack is batch 4 + NUM_WORKERS=0 + correct checkpoint directory.**

Evidence chain: `DataLoader worker killed by signal: Terminated` is the host-side OOM-killer signature (not a CUDA OOM, which raises in-process); the main run already eliminated this class of crash with NUM_WORKERS=0 (113, config flip 5), but the ablation preset kept workers; the 3060 has 25% less VRAM and the ablation used 50% larger batches. Prescription for all future 3060 runs: BATCH_SIZE=4 (accept effective batch 16, matching the main run — which also removes a confound from the ablation comparison), NUM_WORKERS=0, and fix the checkpoint-directory misrouting before launch. If crashes persist at batch 4, the fallback is BF16 autocast (F6, code exists but never run — treat as experimental) or batch 2 with accum 8.

## 7.6 Caveat transparency

**Answer: Full disclosure, compact form. One "Comparability Notes" subsection of ~10 lines.**

The four caveats (POS fill-forward paradigm; split difference pending D1; per-frame vs temporal activity; $299 promotional vs $429 MSRP) each cost one sentence. The asymmetry is stark: a declared limitation costs a sentence and buys credibility; a discovered omission can cost the paper — and the STORM-PSR authors are plausible reviewers for any IndustReal submission. Structure each caveat as *disclosure + quantification + what the honest comparison is*: e.g., "Our decoder's fill-forward constraint guarantees monotone predictions and inflates POS relative to transition-detection methods; we therefore also report the canonical-order blind baseline (POS=X.XX) to bound the contribution of visual evidence." That last clause is why Q43 must run — it converts the biggest vulnerability into a demonstration of rigor.

## 7.7 System configuration for the 8-day run

**Answer: Adequate; three cheap hardenings recommended.**

The memory math is fine (~30+ GB free, training niced +10, crash cause is GPU-side not RAM-side per 111 §2.2). Recommended: (1) close or minimize Chrome on the training box — 3–4 GB RSS and GPU 0 residency for zero benefit; the Xorg+Chrome VRAM on the 3060 also participated in the watchdog pathology (113 §2.5); (2) kill the two idle Claude sessions (~0.9 GB); (3) add a disk-space check to the 1000-step checkpoint hook — crash_recovery.pth writes 738 MB every ~30 min and the project already sits at 26 GB; a full disk mid-epoch is a preventable total-loss failure mode. Optional: cron the `.gpu_heartbeat` age and alert if stale >10 min, so a silent hang costs minutes rather than a night.

## 7.8 3060 priority: T2, ablations, or D-experiments first?

**Answer: (c) D1/D3/D4 first (today), then A2–A4 ablations; T2 only if the T3 gate passes.** Full reasoning in Section 0.2 decisions 1–4. The one-line version: comparability experiments cost hours and unlock the two most attack-prone claims; ablations support the paper's *thesis*; T2 supports its *weakest section* and its value is unknowable until T3 prices it.

## 7.9 The 9 zero-GT detection channels

**Answer: Report both numbers, standard mAP@0.5 as the comparability headline, mAP50_pc as the honest companion, in a single table with the dilution mechanics stated once.**

Never lead with 0.506 alone — reviewers will read it as metric shopping. Never report 0.317 alone — it understates the model by a documented artifact. The defensible presentation: one detection table with three rows (mAP@0.5 = 0.317; mAP50_pc = 0.506; n_present = 15/24) and one sentence: "Nine of 24 ASD states have zero ground-truth instances in the validation subsample; mAP50_pc excludes them; D3 full-set evaluation resolves whether this is a subsample artifact." Then D3 (running today) likely dissolves much of the issue: Q40's hypothesis is that full-set mAP lands 0.33–0.36 with more channels populated. If full-set eval populates all 24 channels, the pc/standard distinction shrinks to a footnote. Also note 112 Anomaly 2 (below) must be fixed first, since the n_present bookkeeping is currently buggy.

## 7.10 Ego-pose position: fix or remove?

**Answer: Remove from claims; publish 6-DoF orientation only. Later, test Q12 (position-loss removal) — which may improve the orientation numbers.**

The position pathway has an unknown unit scale (HEAD_POSE_POS_SCALE=100.0, units undocumented), an explicit "DO NOT USE FOR REPORTING" in evaluate.py:1918-1926, and no published comparison to lose. Fixing it means reverse-engineering the HoloLens coordinate frame under deadline — negative expected value. The interesting twist is Q12: if the noisy position term is *adding gradient noise to the orientation regression*, removing it could take forward MAE from 8.14 toward 7.5–7.8. That is a config-change ablation for the post-D-experiment queue, and if it works, the paper's story improves twice (better number, no caveat).

## 7.11 Realistic single-task ConvNeXt-Tiny detection ceiling

**Answer: Estimate 0.45–0.55 mAP@0.5 on this dataset with proper initialization; the gap decomposition should be stated as pretrain ≳ multi-task cost > head architecture.**

Scaling logic: ConvNeXt-Tiny with a well-tuned dense head reaches ~43 AP@[.5:.95] on COCO with 12-epoch schedules, which corresponds to ~62–65 AP@0.5 — but that is with COCO-scale data (118K images, dense boxes) and detection-tuned recipes. IndustReal offers 4,710 GT frames (24× less box supervision), extreme state-class confusability (1–2 bit ASD differences), and no COCO pretrain of the head. Halving the COCO-relative performance for the supervision gap gives ~0.45–0.55. Implications: (a) the A1 result of 0.184 is *certainly* wrong-as-measured — from-scratch init, misrouted checkpoints, different batch dynamics; rerun A1 correctly before believing any multi-task-cost number; (b) if a corrected A1 lands ~0.45, the paper's decomposition reads "0.838 (YOLOv8m, COCO+synth pretrain, dedicated) → ~0.45 (our backbone, dedicated) → 0.317 (ours, 4-task)" — a clean partition into pretrain/data gap (~0.39) and multi-task cost (~0.13); (c) Q26 (discriminative-LR ImageNet init) and Q38 (YOLOv8m pseudo-labels) are the two levers most likely to move the ceiling itself.

## 7.12 100 vs 200 epochs

**Answer: Keep 100.** OneCycleLR's cosine decay is doing the work in epochs 30–100; extending to 200 either (a) stretches the same schedule, delaying peak learning for no benefit, or (b) appends a near-zero-LR tail where EMA has already converged. With 26K training frames and a 1,769:1 parameter-to-sample ratio, the marginal epochs raise overfitting risk instead of reducing it. If, at epoch ~60, validation metrics are still climbing steeply (unlikely under cosine decay), the correct move is a *second* cycle from the best checkpoint with a fresh OneCycle schedule — a deliberate decision then, not a preemptive 200-epoch commitment now.

## 7.14 Detection: quality, duplicates, or class confusion?

**Answer: The DET_PROBE data points to (c) class confusion as the dominant loss mode, with (b) duplicate suppression as a secondary, fixable contributor. Localization (a) is demonstrably fine.**

Diagnosis from available evidence: 3,814 predictions at IoU>0.5 with max IoU 0.942 means the boxes are landing on the objects — localization is not the problem. The per-class AP profile is the tell: channels differing by 1–2 ASD bits from high-AP neighbors score near zero (ch16=0.000, ch19=0.000, ch22=0.063 despite 28 GT) while visually distinctive states score 0.7–0.94. That is class confusion between adjacent assembly states, exactly what a binary-code taxonomy predicts. NMS interacts with this: when two near-identical states compete, greedy NMS deletes the loser entirely (Q1's mechanism). Prescription: Soft-NMS (Q1) + TTA (Q50) as immediate inference-only tests; per-class confusion matrix extraction from the D3 full eval to confirm the confusion pairs; longer-term, the OHEM/gamma question (Q2/Q5) addresses whether training itself starves the confusable classes.

## 7.15 Is the 0.1× backbone LR appropriate?

**Answer: Keep 0.1× for this run; test higher backbone LR only in a fresh-run ablation (fold into Q26).**

The 0.1× convention exists to protect pretrained features, and the one prior data point (113: ImageNet init with 5e-4 backbone LR regressed −0.02 mAP — catastrophic forgetting) confirms this model is sensitive to backbone LR. Raising it mid-run at epoch 12+, while OneCycle is at peak, is the highest-variance intervention available and risks destabilizing all four heads simultaneously. The correct experiment is Q26's discriminative schedule (backbone 1e-5 for 5 epochs, then 5e-5) in a fresh 15–25-epoch run — that tests "can the backbone adapt faster" without betting the main run on it.

## 7.17 Per-frame activity framing

**Answer: Lead with the "zero-marginal-cost byproduct" framing; report as a baseline, not a headline contribution.**

The most defensible sentence structure: "The multi-task architecture yields per-frame action classification *for free* — 0.7M additional parameters, ~5% of forward FLOPS — establishing the first single-frame baseline on the 69-class verb-grouped protocol (macro-F1 0.110, Top-5 0.398)." This framing (a) preempts "why not temporal?" by making single-frame a cost statement rather than a limitation, (b) uses Top-5 = 0.398 as the supporting number showing the model narrows to the right action family, and (c) leaves clean room for the temporal extension whether or not T2 runs. Do not bury it in an appendix — an honest weak baseline stated confidently is fine; a hidden weak result reads as evasive. Yes, "per-frame action classification" is a recognized formulation (frame-level accuracy is a standard reported metric in temporal action segmentation literature); cite that literature when defining the task.

## 7.18 MViTv2 remap strategy

**Answer: Sum the probabilities (equivalently logsumexp the logits) of merged classes — do not average, do not max.**

If verb-grouping merges fine classes {a, b} into group g, then P(g) = P(a) + P(b) under the law of total probability; averaging under-weights merged groups and max discards mass. Practical protocol for T3: (1) extract the 75→69 mapping from the codebase's verb-grouping table (it is the ground truth of what was merged — do not reconstruct it from class names); (2) softmax the MViTv2 logits first, then sum probabilities within groups; (3) argmax over the 69 groups for Top-1, and compute macro-F1 over groups; (4) sanity-check that ungrouped classes' predictions are bit-identical before/after remap. One warning: the doc-111 arithmetic "75−69=6 means 3 merged pairs" is not guaranteed — 6 fewer classes could be one 7-way merge or several small ones; read the actual mapping.

## 7.20 Validation subsampling (D3 timing)

**Answer: Run D3 immediately on the 3060, before epoch 12's validation completes. Concurrency with the main run is safe.**

Reasons for urgency: (1) every currently-quoted number derives from the 250-batch subsample; if the full-set numbers shift (Q40 hypothesis: mAP50 → 0.33–0.36), every doc and draft table needs one coherent update, and earlier is cheaper; (2) D3 doubles as the first real-GPU verification of F22/F22b (the PSR eval fixes verified only on CPU synthetic data — 113 §1.3), which gates the POS=0.968 claim; (3) best-model selection is being made on subsampled combined metrics — if full-set ranking differs, better to know at epoch 12 than epoch 60. Concurrency: D3 is inference-only on the 3060 with its own CUDA context; the only shared resources are disk read bandwidth (val images are not in the main run's RAM cache — schedule D3's heavy read phase during main-run compute, which is trivially satisfied since the main run is compute-bound at 55% GPU-util with NUM_WORKERS=0) and CPU (pin D3 to fewer threads, `OMP_NUM_THREADS=4`, and nice it below the training process).

## 7.21 Overfitting risk at 100 epochs / 26K samples

**Answer: Moderate risk, adequately mitigated; take option (c), which is already in place, plus one addition.**

(a) Reducing to 50 epochs sacrifices the cosine-decay refinement phase — reject. (b) Stronger augmentation mid-run changes the data distribution under a live schedule — reject for this run; queue color jitter + small-angle rotation (which also serves ego-pose per Q14) for fresh runs. (c) EMA at 0.995 is the right mechanism and is enabled — the EMA model should be the *evaluated and published* model; verify the eval path actually uses EMA weights (worth one log-line check). Addition: VAL_EVERY=1 now provides per-epoch curves — instrument a simple divergence monitor (train loss ↓ while val combined ↓ for 5 consecutive epochs ⇒ early-stop consideration at the best checkpoint). The honest overfitting signal to watch is per-class detection AP on low-GT classes, which overfit first.

## 7.23 Venue selection

**Answer: (A) dual track.** Covered in 7.4 and Section 0.2 decision 9. Rejecting (B): it forfeits a nearly-free publication and time-stamps nothing until mid-2027. Rejecting (C): ICHCIIS is an HCI venue — the full multi-task ML paper is a fit mismatch there, and burning the complete result on the lower-prestige venue inverts the value ordering. Content split discipline: the ICHCIIS paper gets ego-pose + per-frame activity + PSR-POS with the operator-monitoring/HCI framing; the AAIML paper gets the architecture, ablations, comparability suite, and efficiency thesis. Overlap limited to the dataset description and the ego-pose definition, with the ICHCIIS paper cited.

## 7.24 The actual deadline

**Answer: Yes — ICHCIIS-26 abstract (July 15) with Track A + D1/D3/D4 results; AAIML 2027 (Jan–Feb) with everything.** The arithmetic works: D-experiments finish July 4–5; the abstract needs headline numbers, not the full paper; the ego-pose/POS/per-frame numbers are already final pending D3 confirmation. Do not gate the abstract on main-run completion (July 12–16) — epoch-11 numbers with "results at epoch 11 of 100; final numbers in camera-ready" is standard and safe, since every trend is upward.

## 7.25 The 50-question file

**Answer: Acknowledged as the priority queue; all 50 answered in Section 7 of this document, with a consolidated execution ordering in Section 8.**

---

# Section 2: Answers to Document 112 — The Seven Anomalies

## Anomaly 1: Validation loss rising while validation metrics improve

**Disposition: Expected and benign; the doc's own hypothesis is correct with one addition. No action.**

The validation loss is Kendall-weighted, and the log_var trajectory (112 Appendix D) shows weights actively shifting toward the noisy activity head across exactly the epochs in question. Additionally, the regularization term Σ·log_var itself changes the loss floor as log_vars move. A Kendall-weighted loss is not a model-quality metric and should never be used for model selection — the combined metric already fills that role. Recommendation: log the *unweighted* per-head validation losses alongside, so quality trends are visible without Kendall confounding.

## Anomaly 2: det_n_present_classes=0 in all RF4 validations

**Disposition: Real bug, must fix before publication. Priority: this week.**

The state is internally contradictory (mAP50_pc=0.5063 cannot be computed with zero present classes) and it touches the best-model selection branch. Most likely a dict-key mismatch between the eval return and the logging/selection code (the metric was added in the det_mAP50_pc fix wave; the n_present key plumbing was likely missed on one path). Action: trace the key from evaluate.py's return dict through train.py's Val-line formatting and the combined-metric branch; add an assertion `(n_present == 0) == (mAP50_pc in (0, nan))`. Verify which branch actually selected best.pth — 112 §11 suggests the post-val path used the pc value regardless, so the damage is likely logging-only, but that must be confirmed, not assumed.

## Anomaly 3: Epoch 7–8 training loss spike

**Disposition: Explained by Kendall re-weighting, not by validation side-effects. No action.**

The spike (2.49 → 3.02 → 3.27) coincides exactly with act_log_var swinging from −0.008 to +0.205 (epoch 7→8) and the raw activity loss spiking 1.244 → 1.767 — the same epochs as the documented activity-head collapse-and-recovery (macro-F1 0.097 → 0.049 → 0.110). A Kendall-weighted total loss is not comparable across epochs when log_vars are moving; the per-head raw losses are the valid comparison, and detection's raw loss declined monotonically straight through the "spike." The val-triggered-state-change hypothesis is unnecessary.

## Anomaly 4: Ablation det-mAP lower than multi-task (0.184 vs 0.317)

**Disposition: Measurement artifact — do not interpret, rerun. The current A1 is triple-confounded and its number should not appear anywhere.**

Confounds, in order of severity: (1) initialization — the ablation trained from scratch while the multi-task number reflects a checkpoint lineage with more accumulated backbone training; (2) checkpoint misrouting into `full_multi_task_tma_tbank/`, meaning resumes may have loaded state from a different run's lineage — this alone invalidates the run; (3) batch 6 + peak_factor differences + 3060 crash-restarts. Correct A1 protocol: same init policy as the main run, same batch size (4/accum 4), own clean checkpoint directory, NUM_WORKERS=0, 25 epochs. Until then, the paper has no valid multi-task-cost number, which is why A2–A4 (and a redone A1) outrank T2 on the 3060.

## Anomaly 5: Phase A/B/C combined metric formula unknown

**Disposition: Do not spend time reconstructing it. Quarantine those numbers.**

Phase A/B/C predates multiple correctness fixes (F18, F22/F22b among them), so its metrics are non-comparable to RF4 regardless of formula. Mark the era "historical, pre-fix, non-comparable" in all docs; never mix its combined values into any trajectory plot. The only Phase A/B/C fact worth carrying forward is qualitative: the architecture trained end-to-end without NaNs.

## Anomaly 6: Activity metrics zero until epoch 11

**Disposition: Consistent with the F18 double-ramp fix timeline plus threshold effects; verify with one cheap check.**

F18 (activity ramp was ramp², i.e., 4% effective weight at epoch 0 instead of 20%) landed ~epoch 5–6; 5 epochs of post-fix training to reach measurable macro-F1 on a 69-class imbalanced problem is plausible. The cheap verification: run the current eval on the epoch-8 checkpoint *with today's eval code*. If epoch-8 activity metrics are now nonzero, the earlier zeros were partly an eval-code artifact (F22-adjacent); if still ~zero, the head genuinely wasn't producing signal and the F18 explanation stands alone. This matters for the paper's training-dynamics narrative and costs ~30 minutes on the 3060.

## Anomaly 7: PSR F1 +332% (epoch 8→11) while POS flat

**Disposition: Expected metric structure, not an anomaly — and it is the strongest evidence for the paper's "detection is the PSR bottleneck" claim.**

POS is edit-distance-based over the *order* of transitions; the fill-forward decoder produces valid monotone orderings almost immediately, so POS saturates from the first epoch the decoder works (0.966 at epoch 8). F1@±3 requires transitions at the *right time*, which depends on the quality of the detection-derived s2 features — so F1 tracks detection improvement (mAP 0.208 → 0.317 over the same epochs) with leverage. Projection: F1 should continue climbing roughly with detection quality; POS will stay ~0.97 forever. In the paper, present POS and F1 as measuring order-correctness vs timing-precision respectively — this reframes the odd-looking pair (0.968 / 0.144) as informative decomposition rather than inconsistency.

---

# Section 3: Answers to Document 113 — Untested-Fix Triage

Ranked by (risk to published numbers) × (cost to verify). The first three gate publication.

| Rank | Item | Why it gates / verdict | Verification | Cost |
|------|------|------------------------|--------------|------|
| 1 | **F22/F22b GPU-path** (PSR eval fixes, CPU-synthetic-verified only) | POS=0.968 is the flagship claim; if the GPU eval path mis-shapes tensors, the headline number is wrong | D3 full eval on 3060 + assert decoder I/O shapes + spot-check 3 sequences by hand against GT | Free (inside D3) |
| 2 | **Full 38K eval (EVAL_MAX_BATCHES=0)** | Every published number currently comes from a 2.6% subsample | This *is* D3 | 1h |
| 3 | **YOLOv8m weights availability (D1/D4 dependency)** | Two of three P0 experiments die if the repo weights are gone or class-order mismatched | Download now, verify class mapping on 10 frames before the full run | 30 min |
| 4 | **Multi-seed variance (Q15)** | Single-seed numbers with no error bars are the most common AAIML-tier review complaint; ego-pose (claimed to 3 significant figures) is most exposed | Two additional 25-epoch runs (seeds 7, 123) on the 3060 *after* ablations; report mean±std for headline metrics; full-length reruns unnecessary — 25-epoch relative variance suffices for error bars | 3–4 days, deferrable to August |
| 5 | **Ablation presets A2–A4 (F16, never run)** | Efficiency thesis currently unsupported (Anomaly 4) | Run per Section 5 schedule, after preset audit (batch, checkpoint dir, init policy) | 5 days |
| 6 | **E2 PSR tau (not implemented)** | Missing SOTA metric weakens PSR table, but per-frame tau is paradigm-different anyway (Q44) | Implement per-component delay measurement; fold into Q17 which needs the same computation | 1 day |
| 7 | **BF16 autocast (F6, never run)** | Pure upside option (2× throughput) but unproven with FocalLoss | Test on 3060 in a 2-epoch smoke run *only if* 3060 capacity becomes the binding constraint; never flip it on the main run mid-flight | 0.5 day, optional |
| 8 | **grad_cosine_probe (F12, never run)** | Feeds Q23 (PCGrad decision); diagnostic only | Run once on the main process at a probe step; zero training impact | 1h |
| 9 | **Assert-and-crash mode / LION optimizer** | Debug/experimental; no path to the paper | Skip | — |

One structural recommendation: 113 lists 38+ fixes of which the deepest (F1 seq-batch gradient wipe — ~80% of backbone signal silently lost; F13 — all probes structurally never firing; F18 — ramp²; F22b — constraint never applied) share a single root cause: **silent failure with no assertion**. The paper's training-pathologies section (116 already plans one) should name this pattern explicitly — it is a genuinely useful engineering contribution and reviewers of systems papers reward it. Going forward, every fix of the form "X was silently not happening" should land with a runtime assertion that X is happening.

---

# Section 4: Answers to Document 114 — Final Comparability Rulings

## 4.1 Per-metric rulings (consolidating 114 §5–7 with decisions)

**Category 1 — claim now, as-is:**
- **Ego-pose fwd/up MAE (8.14/7.06°):** Claim as first baseline. Guard rails: never compare to face-based head-pose literature (OpenFace/6DRepNet — different task); state the HL2 sensor-noise context (~5–7°) as an approximate floor, cited as an estimate, not a measurement. Verify epoch-stability (the number should be quoted from the *final* best checkpoint with mean±std once Q15 runs).
- **PSR POS (0.968):** Claim with the two-part disclosure (paradigm + canonical-order baseline from Q43). Do not publish before Q43 runs — the blind-baseline number belongs in the same table row.
- **mAP50_pc (0.506):** Claim as honest companion metric, never as the headline (ruling in §1, 7.9).
- **PSR edit (0.752) and component accuracy (0.346):** Supplementary table only.
- **Per-frame activity (0.110 macro-F1, 0.398 Top-5):** Claim under the renamed task with the zero-marginal-cost framing (7.17).

**Category 2 — claim after experiment:**
- **Detection mAP@0.5 vs YOLOv8m:** After D1. Write the comparison sentence conditionally until then. If D1 lands 0.75–0.85 the published split-compatibility holds; if <0.70, the honest sentence becomes "our split is harder than the published protocol; on our split the gap is X" — which *helps* us.
- **PSR F1:** After D4 + Q17 decomposition (§1, 7.3).
- **Temporal activity:** Only if the T3 gate passes (§1, 7.2-second).
- **Retrieval F1@1/MAP@R (R1):** Optional, P2. Positive-sum if it lands ≥20 ("detection-supervised embeddings are competitive with contrastive training"), and quietly droppable if not. Run only after all P0/P1 work; do not let it displace ablations.

**Category 3 — never comparable; related-work only:**
- Paper 3 retrieval metrics vs our detection (different task), MViTv2 native 75-class Top-1 (different protocol, pretrain, modality), PSR tau vs transition-detection tau (different paradigm — report ours, labeled per-frame, alongside theirs with the difference stated; never in the same column unannotated).

## 4.2 The comparison-table architecture for the AAIML paper

Three tables, not one — mixing comparability categories in a single table is how reviewers conclude the comparisons are unfair:
1. **Table "Prior art on IndustReal"** — published numbers verbatim (YOLOv8m 0.838; MViTv2 65.25%; B3 0.797/0.883/22.4s; STORM 0.812/0.901/15.5s), with their protocols summarized.
2. **Table "Direct comparisons"** — only rows where protocol is matched: POS (with paradigm note + Q43 baseline), detection after D1, PSR F1 after D4.
3. **Table "Original baselines"** — ego-pose, mAP50_pc, per-frame activity, component accuracy: no SOTA column at all, which is the point.

---

# Section 5: Answers to Document 115 — Revised Execution Plan

## 5.1 Revised GPU allocation (deltas from 115 §7 in bold)

```
Day 0   (Jul 4):  5060Ti: main epoch 12+          3060: D1 (2h) → D3 (1h) → D4 (2-3h)
                                                   **+ Q43 canonical POS (CPU, hrs)**
                                                   **+ Q50 TTA & Q1 Soft-NMS on epoch-11 ckpt (2-3h)**
Day 1   (Jul 5):  5060Ti: main                     3060: **Q17 tau + Q18 thresholds (1 day, inference)**
                                                   **+ T4 act_top1 (1h) + T3 MViTv2 remap start**
Day 2   (Jul 6):  5060Ti: main                     3060: T3 completes → **T2 GO/NO-GO GATE**
Day 3-7 (Jul 7-11): 5060Ti: main                   3060: **A1-redo + A2-A4 ablations (batch 4, clean dirs)**
                                                        (if T3 gate passed: T2 displaces A3/A4 to week 3)
Day 8+  (Jul 12-16): 5060Ti: main completes        3060: finish ablations / B1
Week 3:            5060Ti: B1/C1/E1 + (T2 if gated-in and not yet run) + Q15 seeds
Jul 15:            ICHCIIS-26 abstract — needs only Track A + D1/D3/D4, all done by Day 1
```

## 5.2 The five go/no-go gates

| Gate | When | Condition | If fail |
|------|------|-----------|---------|
| G1: T2 launch | After T3 (~Jul 6) | Remapped MViTv2 macro-F1 ≤ ~0.20 | Skip T2; adopt per-frame framing (Option C); 3–4 GPU-days saved for ablations/seeds |
| G2: OHEM ablation | Epoch ~30 (~Jul 7) | mAP50_pc < 0.55 OR cls_mean < −9.5 with flat mAP | Launch Q5 on 3060 (25 epochs), concurrent with main run |
| G3: PSR narrative | After D4+Q17 (~Jul 6) | D4 F1 ≥ 0.45 and tau mostly within ±3 | If fail: switch PSR story to paradigm-limited; lean on Q18/Q19 fixes and per-frame tau (Q44) |
| G4: POS claim | After D3+Q43 | GPU-path PSR eval verified AND blind baseline ≤ ~0.90 | If blind baseline > 0.93: demote POS from headline to supporting result; the paper survives on the other six contributions |
| G5: abstract submit | Jul 13 | D1/D3 numbers in hand (they will be) | Submit with epoch-11 numbers + "preliminary" framing regardless |

## 5.3 Risk-register deltas (against 115 §8)

Three risks under-weighted in 115: (1) **disk exhaustion** — 738 MB × every-1000-steps on a 26 GB-and-growing tree; add the free-space check (7.7); (2) **eval-path/EMA mismatch** — confirm published numbers come from the EMA weights, one log check; (3) **single-seed exposure** — no current plan line for Q15; slot seeds 7/123 (25-epoch) into week 3 on the 3060. One risk over-weighted: "ICHCIIS deadline missed" — with the revised Day-0/Day-1 plan, everything the abstract needs exists a week early.

---

# Section 6: Answers to Document 116 — Paper Synthesis Verdicts

## 6.1 The seven contributions, ranked by survivability under review

1. **C1 Ego-pose first baseline — anchor contribution.** Zero caveats. Lead the paper with it alongside C2.
2. **C2 Single-GPU multi-task system — the thesis.** Survives *only* with corrected ablations (A1-redo, A2–A4) and E1 FPS measured (the current 4.8 FPS is a LaTeX estimate — never print an estimated FPS in an efficiency paper; E1 is one hour). Price framing: lead with $429 MSRP, mention $299 street — "a sub-$450 consumer GPU" is unattackable, "$299" invites a footnote war for zero gain. Recommend retitling accordingly.
3. **C5 PSR POS beats SOTA — strong but conditional** on G4 (Q43 blind baseline + GPU-path eval verification). With both disclosures, it is rigorous; without Q43, it is the paper's most attackable claim.
4. **C3 Honest present-class mAP — keep, as metric-design contribution,** presented as the companion-metric pattern (7.9), explicitly not as the headline.
5. **C4 Per-frame action classification — keep as baseline** with the zero-marginal-cost framing (7.17). Do not oversell; one subsection.
6. **C7 Temporal activity under verb-grouped protocol — conditional on G1.** If T2 is skipped, this contribution is simply dropped and C4 absorbs the activity story; the paper loses little.
7. **C6 Non-contrastive embedding baseline (R1) — optional garnish.** Include only if R1 runs and lands ≥ F1@1≈20. Never let it displace ablation time.

## 6.2 Fallback narrative (no further experiments at all)

116 §10's five-claim fallback is confirmed viable with one amendment: even the zero-GPU fallback should include Q43 and the TTA/Soft-NMS numbers, since they cost hours and require no training. Minimal viable paper = C1 + C2(params-only, FPS measured) + C3 + C4 + C5(with disclosures). That is an honest workshop-to-mid-tier paper. Every experiment that completes moves one row from "original baseline" to "direct comparison," which is the difference between mid-tier and strong-accept at AAIML.

## 6.3 The press-release / abstract sentence

The strongest defensible one-liner, post-D-experiments: *"A single 46.5M-parameter model on one consumer GPU performs all four IndustReal tasks simultaneously — establishing the first ego-pose baseline (8.1° forward MAE), exceeding published procedure-order SOTA (POS 0.968 vs 0.812, per-frame paradigm), and retaining X% of dedicated-pipeline detection quality at 67% fewer parameters."* Fill X from D1+A1-redo.

---

# Section 7: Answers to Document 117 — All 50 Questions

Verdict key: **T0** = execute now (inference-only or config-only, this week) · **T1** = execute before paper submission · **T2** = gated / conditional · **SKIP** = do not execute (with reason). Each answer states the verdict, the expected outcome relative to the question's hypothesis, and the reasoning.

## Category 1 — Detection (Q1–Q5)

**Q1 Soft-NMS — T0.** Hypothesis endorsed with moderated magnitude. The ASD taxonomy's 1–2-bit inter-class proximity is precisely the regime where greedy NMS deletes correct-but-second-place states; channels 16/19/22's near-zero AP despite adequate GT is the predicted signature (see 7.14). Expect +0.02–0.05 mAP50_pc, concentrated on the transitional channels; the +0.10–0.20 per-channel upper estimates are optimistic because confusion happens at the classifier too, not only at suppression. 30-minute change, inference-only, zero risk: run it against epoch-11 today, and re-run inside D3 so the published numbers include it (with the NMS variant disclosed).

**Q2 OHEM min_neg 32→8 — T2 (fold into Q5's gate G2).** The mechanism is real (a 32-negative floor on 1–5-positive frames is an aggressive ratio), but do not spend a separate 25-epoch ablation on min_neg alone. If G2 triggers, run a single 3-arm ablation: {OHEM off + γ_neg=2.0 (Q5)}, {min_neg=8 (Q2)}, {baseline} — one experiment, both questions answered. If G2 never triggers (detection keeps climbing), neither is needed for the paper.

**Q3 BiFPN — SKIP for this paper.** Expected +0.02–0.04 is real but costs a fresh 25-epoch run plus architecture churn late in the cycle, and it changes the model the whole comparability suite was measured on. Correct home: the journal extension. Note the stated rationale ("FPN was grad-starved for 11 epochs") argues for *re-training time with the fixed FPN* — which the main run is already delivering — more than for a new neck.

**Q4 Head depth 2×256 — SKIP for this paper.** The overfitting logic is plausible but the payoff (±0.02–0.05, sign uncertain) does not justify a fresh run now. Fold into the same future sweep as Q3. If free 3060 time appears in week 3, this is the first optional detection ablation to run, because "we halved the head and nothing changed" is itself a nice efficiency datapoint for C2.

**Q5 OHEM-off + γ_neg=2.0 — T2, gated on G2 (epoch ~30).** This is the definitive test of the project's longest-standing hypothesis (OHEM+Focal double-suppression). Endorse the design; add per-class AP and cls_mean tracking so the result is diagnostic either way. If it recovers channels 16/19 to >0.05 AP, it is a Training Pathologies section headline; if not, the double-suppression theory is finally retired — also valuable.

## Category 2 — Activity (Q6–Q10)

**Q6 75 vs 69 classes — SKIP as experiment; answer via T3 instead.** The 5–6-day cost buys at most +0.01–0.03 macro-F1 and a protocol change late in the game. T3's remap machinery already quantifies what grouping does to the *comparison*; the paper reports the 69-class protocol with the mapping published. Reverting to 75 would also un-do the comparability groundwork already laid.

**Q7 TCN 4-layer dilations — T2, bundled into T2-the-experiment if G1 passes.** Endorsed: if the temporal head runs at all, run it with receptive field ≥ the mean action length (31 frames ≥ 19) from the start. This is a design correction to T2, not a separate ablation — do not run 2-layer first and 4-layer second; there is no budget for two temporal runs.

**Q8 Attention pooling — T2, same bundle as Q7.** <100 params, <10 lines, plausible +0.01–0.03: include it in the T2 configuration by default. Not worth any standalone run.

**Q9 ACTIVITY_GRAD_BLEND_RATIO 1.0→2.0 — T1, but on the 3060 in a 5-epoch probe first, NOT on the live main run.** The env-override makes it tempting to apply mid-training, but the epoch 5–8 history (activity collapse coinciding with detection's LR-peak takeover) shows this coupling is the system's most delicate equilibrium, and the main run is currently *improving* on all fronts — do not perturb a winning run. Protocol: resume epoch-11 checkpoint on the 3060, 5 epochs at ratio 2.0, compare. If activity gains ≥0.02 with detection flat, apply to the main run at the next epoch boundary.

**Q10 DET_GT_FRAME_FRACTION 1.0 — SKIP in this form; see Q49.** Q10 (fraction→1.0, i.e., natural sampling) and Q49 (fraction→0.60) pull the same knob in opposite directions, and they cannot both be right. The sampler-imbalance warning (7.4× activity distortion) argues the current 0.40 is already a compromise. Ruling: keep 0.40 for the main run; test the knob once, post-hoc, in a 2-arm 5-epoch probe {0.60, 1.0} on the 3060 in week 3 if capacity allows. Expected outcome: 0.60 helps detection slightly at negligible activity cost (Q49's direction), because detection is the scarcer supervision — but this is exactly why it must be measured, not asserted.

## Category 3 — Ego-pose (Q11–Q15)

**Q11 Geodesic loss — T1 (fresh-run ablation, week 3).** Endorsed; the gradient-vanishing argument for MSE-on-unit-vectors at small angles is textbook-correct, and ego-pose is the contribution most worth polishing (sub-7.5° strengthens C1 materially). 25 epochs on the 3060. Combine factorially with Q12 (2×2 is overkill — run {geodesic + no-position} vs baseline, the combined best-guess config, and only decompose if it fails).

**Q12 Position-loss removal — T1, same run as Q11.** Double win if it works (better orientation + caveat deleted). See 7.10.

**Q13 FiLM near-identity check — T0 (the checkpoint inspection), T2 (the removal retrain).** The inspection is 20 minutes with the epoch-11 checkpoint on CPU: histogram the gammas/betas. If near-identity, do NOT retrain now — just note it and remove FiLM in fresh runs. The 400K params matter for the params table footnote, not for performance.

**Q14 Rotation augmentation — T2, week-3 bundle.** Plausible but second-order vs Q11/Q12; the ±15° rotation also changes detection/PSR inputs, so it is not a pose-only intervention — test only in the pose-only ablation context (A2), where it is clean.

**Q15 Multi-seed variance — T1, non-negotiable for the AAIML paper.** Reviewers at any credible venue will ask for error bars, and ego-pose is quoted to 3 significant figures from one seed. Two additional 25-epoch runs (seeds 7, 123) suffice for std estimates on all headline metrics; schedule week 3 on the 3060. The ICHCIIS short paper may go out single-seed with a stated limitation.

## Category 4 — PSR (Q16–Q20)

**Q16 D4 — T0 (it is P0; run today).** Covered in 7.3. Expected F1 0.45–0.65; interpret jointly with Q17.

**Q17 Tau distribution — T0 (run BEFORE or with D4).** The decomposition key for the entire PSR narrative (gate G3). Hours of work on existing predictions; also delivers 80% of E2.

**Q18 Per-component thresholds — T0.** Inference-only, grid-search on validation, expected F1 0.144 → 0.17–0.22 with POS roughly held. One methodological requirement: tune thresholds on a *held-out portion* of validation (or by cross-validation across recordings) and report on the rest — otherwise it is threshold-fitting on the test set and a reviewer will correctly object. The prevalence-aligned prior (rare components get low thresholds) makes this defensible as a principled calibration, not a hack.

**Q19 Temporal smoothing 0.20 — T1 (5-epoch probe on 3060, week 2).** Cheap, plausible +0.02–0.05 F1; the over-smoothing failure mode at 0.50 predicted by the hypothesis is right, so sweep {0.10, 0.20, 0.35} in one probe.

**Q20 Seq freq 4→2 — T2.** The DETACH_PSR_FPN protection makes it safe in principle, but seq batches at freq 2 halve the effective throughput of all *other* heads' supervision (seq batches zero the other losses, per 111 §3.7). Since detection is the current bottleneck and PSR-F1's bottleneck is detection quality (Anomaly 7), doubling PSR's signal at detection's expense is probably net-negative *now*. Revisit only if, post-D4, the paradigm (not detection) proves to be the F1 limiter.

## Category 5 — Multi-task balancing (Q21–Q25)

**Q21 Kendall vs fixed at equilibrium — T1 (this is B1; run it, 2 days, week 3).** The equilibrium analysis (det and act still 0.3–0.6 from equilibrium ⇒ Kendall still actively steering) supports keeping Kendall for the main run and predicts B1 shows a modest Kendall win. B1 matters less as a performance question than as a reviewer-defense: "we compared learned vs fixed weighting" closes an obvious question.

**Q22 GradNorm — SKIP.** Its predicted trade (activity +0.03, detection −0.02, PSR −0.03) is a net loss given detection is the binding constraint, and swapping the balancing algorithm invalidates the fix-stack's hard-won Kendall diagnostics (HP_PREC_CAP, log_var logging, clamp interpretations). One balancing framework per paper.

**Q23 PCGrad / gradient cosine — T0 for the measurement, T2 for the method.** The F12 probe exists and has never run (113 §1.3): fire it once, get the det-pose cosine for free. If cos < −0.3, *report the conflict measurement in the paper* (it is an interesting multi-task datapoint) but adopting PCGrad is a week-3-at-earliest decision — same reasoning as Q22, do not churn the balancing stack mid-run.

**Q24 HP_PREC_CAP removal — SKIP for the main run; T2 as a 5-epoch probe if curiosity persists.** The cap exists because uncapped head-pose demonstrably took over the backbone (54.6× precision). The hypothesis that *some* extra pose gradient would help detection via orientation cues is not crazy, but the downside (re-triggering the documented takeover) is a known catastrophic mode. If tested at all: 5-epoch probe from epoch-11 checkpoint on the 3060, watch detection like a hawk. A softer intermediate (cap at 2× det precision instead of 1×) is the version worth trying.

**Q25 Log_var symmetric init — SKIP.** It answers a convergence-speed question (reach 0.363 at epoch 7 vs 11) that no longer matters — the run is past that region and no fresh full run is planned where 3 epochs of savings would pay for the experiment. Apply all-zeros init as a default in fresh runs (it is the standard init anyway) without a dedicated study.

## Category 6 — Architecture (Q26–Q30)

**Q26 ImageNet pretrain with discriminative LR — T1, the highest-value architecture experiment.** The prior −0.02 result is exactly what catastrophic forgetting from 5e-4-on-pretrained-weights produces, so the question is genuinely open and the upside (+0.03–0.05 mAP plus faster convergence for every future run) compounds. 15-epoch run on the 3060, week 2–3, backbone 1e-5→5e-5 staged. If it wins, it becomes the base config for T2/A2–A4 and the journal extension.

**Q27 Swin-T — SKIP for this paper.** Backbone swaps reset every number and the comparability suite; +0.02–0.05 expected is not worth re-running the world. Journal extension material, alongside Q3/Q4.

**Q28 ConvNeXt-S — SKIP for this paper, and note it cuts against the thesis.** The paper's identity is the *smallest* backbone doing 4 tasks; reaching for a bigger backbone to close the YOLOv8m gap trades the efficiency story for a fraction of the gap. If a scale ablation is ever run, its purpose is the opposite: show Tiny is on the pareto frontier.

**Q29 EfficientNet-B4 — SKIP.** A throughput-parity question with no path to any paper claim; TTA/BF16 are better throughput levers.

**Q30 Detachment ablation (freeze act/pose gradients into backbone) — T2, interesting but redundant with A-suite.** A2–A4 + A1-redo already decompose multi-task cost per head more cleanly (full single-task training vs gradient-freezing mid-lineage). Skip unless the A-suite is somehow blocked; then Q30 is the cheap substitute (env-flag change on a resumed checkpoint).

## Category 7 — Training strategy (Q31–Q35)

**Q31 Peak factor 0.75 — SKIP.** Convergence-speed question, moot for the live run (already past peak), and the NaN risk at higher peaks is documented in the F4 history. Fresh runs keep the F21 auto-factor.

**Q32 EMA 0.999 — T2, week-3 probe only.** Plausible +0.01, but EMA decay interacts with remaining-epoch count (0.999 has a ~1000-step horizon; with ~45K steps left either works). Cheapest honest test: at main-run completion, compare EMA-0.995 weights vs a re-averaged SWA/longer-horizon snapshot offline (Q34 machinery) rather than re-running training.

**Q33 Mixup revisited — SKIP for the main run; optional flag in fresh ablation runs.** The confound story (old anti-mixup result predates the F1 gradient-wipe fix) is credible, but mixup with dense detection targets is genuinely awkward (box mixing semantics), and the expected +0.01–0.02 doesn't buy the risk on the run that matters. If A3 (activity-only) runs, enable mixup there — activity-only is where mixup is clean and most likely to help.

**Q34 SWA — T1-lite (offline, free).** Do the zero-cost version: checkpoint-average epochs ~75–100 of the completed main run offline and evaluate once. If it beats the EMA model by ≥0.01 combined, publish the SWA-selected weights. No training-time change; this is an hour of work after July 12–16.

**Q35 Label smoothing 0.05 — T2, bundle into the Q9 activity probe.** Same 5-epoch probe run can carry {blend 2.0} × {smoothing 0.05} — two knobs, one resumed run, read the interaction. Predicted trade (higher F1, lower pred_distinct at 0.0) means keep ≥0.05.

## Category 8 — Data strategy (Q36–Q40)

**Q36 Per-component weighted PSR loss — T1 (config-level, cheap, targets the weakest sub-heads).** The liveness data (h4/h7–h9 gradient RMS <0.005 through epoch 12 — 112 Appendix B) shows rare components are barely learning; inverse-prevalence weighting is the standard first fix. Expected F1 +0.03–0.07 with a small POS cost (fill-forward becomes slightly less trigger-happy — fine, POS has 0.15 of headroom over SOTA). 5-epoch probe on 3060, apply to main run at an epoch boundary if it wins. Sequence *after* Q18 — thresholds may capture most of the same gain for free.

**Q37 Unity synthetic 50K — SKIP for AAIML timeline.** Real expected value (+0.04–0.07 pc) but the pipeline cost (Unity Perception setup, domain randomization, ingestion) is weeks, not days. Journal extension. Note Paper 1's best YOLOv8m already demonstrates the synthetic-data effect on this dataset — cite it rather than reproduce it.

**Q38 YOLOv8m pseudo-labels — T1, the best detection-data lever available.** Once D1 puts working YOLOv8m inference in hand, pseudo-labeling the 82% non-GT frames is ~a day of plumbing. Two integrity requirements: (a) confidence ≥0.7 as hypothesized, and (b) *the paper must disclose that detection supervision is partially distilled from YOLOv8m* — this changes the detection claim from "trained on IndustReal GT" to "trained with YOLOv8m distillation," which is honest and still supports the efficiency thesis (deployment cost is what the thesis is about, not training provenance). Expected +0.03–0.06 mAP. Run as a 10-epoch branch on the 3060 first; only merge into the main lineage if clean.

**Q39 Active learning 1000 frames — SKIP.** It requires new *annotation* (the dataset's labels are fixed — where would the labels for the 1000 frames come from?). As posed it is really a pseudo-label variant, and Q38 dominates it.

**Q40 Full eval (D3) — T0, running today.** Covered in 7.20. Additional instruction: persist per-frame predictions from D3 to disk, so Q17/Q18/Q43/Q44 and the confusion-matrix analysis all run offline against one artifact instead of re-running inference four times.

## Category 9 — Comparability (Q41–Q45)

**Q41 D1 — T0, the single highest-impact experiment in the queue.** Expected 0.78–0.82 per hypothesis; either outcome is good for the paper (§4.1). Do the 10-frame class-mapping sanity check before the full run (§3, rank 3).

**Q42 act_top1 (T4) — T0, 1 hour.** Also resolves the act_clip (0.0625) vs act_frame (0.177) ambiguity that currently makes the activity table confusing. Expected per-frame Top-1 ≈ 0.15–0.22 per hypothesis — quote *that* as Top-1, never the clip-vote number, with the definition stated.

**Q43 Canonical-order POS baseline — T0, CPU-only, gates the flagship claim (G4).** The most important cheap experiment in the whole set. If the blind baseline scores 0.85–0.93 as hypothesized, the paper reports "POS 0.968 vs blind-canonical X vs SOTA 0.812" and the claim survives with proper context; if >0.93, POS is demoted per G4. Either way the paper is more honest with the number than without it — and it is far better that we compute it than that a reviewer does.

**Q44 Per-frame tau (E2) — T1, mostly delivered by Q17.** Report per-frame tau in its own labeled column; if the hypothesis (0.5–1.5s vs B3's 22.4s) holds, it is a genuinely striking latency result *of the paradigm* — present it as a paradigm property with the trade-off (lower F1 timing precision) stated in the same breath.

**Q45 MViTv2 remap (T3) — T0/T1, runs Day 1–2, feeds gate G1.** Use sum-of-probabilities remapping (7.18). Its dual role: comparability groundwork AND the pricing signal for T2.

## Category 10 — Wildcards (Q46–Q50)

**Q46 Cross-modal FiLM sharing (pose→activity) — T2, contingent on Q13.** If Q13 finds the FiLM near-identity, there is nothing worth sharing and Q46 dies for free. If FiLM is genuinely modulating, this is a nice fresh-run idea (head orientation as an action prior is cognitively plausible) — journal extension tier, not AAIML tier.

**Q47 FeatureBank GRU — T1-investigate (30 min), T2-enable.** First check whether it is actually disabled and what the in-place-grad fix (113: FeatureBank fix, commit 8207632) implies about why. If enabling is config-only, it is the cheapest temporal-context injection available (expected act +0.02–0.06) and could make the *per-frame* number respectable without T2 — test in the same 5-epoch activity probe as Q9/Q35. Caution: it blurs the "per-frame" task definition; if enabled for published numbers, the task name must become "streaming" not "per-frame."

**Q48 MAE pretraining on 188K frames — SKIP for AAIML; first-priority for the journal extension.** The idea is sound (the pretrain_mae.py scaffold even exists) but 50 pretraining epochs + full retrain does not fit any pre-submission window, and it resets all numbers. It is also the single most promising route to closing the detection gap structurally (+0.03–0.06 hypothesized), so it leads the future-work section.

**Q49 DET_GT_FRAME_FRACTION 0.60 — T2, merged with Q10 into one 2-arm probe (see Q10).**

**Q50 TTA — T0, today, the best effort-to-impact ratio in the entire document set.** Endorse with one caution: TTA's 0.03–0.07 gain must be reported as a *deployment-mode* result with the 3–6× inference-cost multiple disclosed, and the FPS table must show both modes (TTA cuts the FPS claim by the same multiple — do not let the two headline numbers quietly come from different modes). Multi-scale {0.8, 1.0, 1.2} × h-flip, merged with Soft-NMS (Q1) since they compose.

---

# Section 8: Consolidated Priority Queue

## T0 — Execute now (Jul 4–6; ~2 days of engineer time, <1 GPU-day, all on 3060/CPU)

| # | Item | Source | Cost | Gates/unlocks |
|---|------|--------|------|---------------|
| 1 | D1 YOLOv8m eval (after 10-frame mapping check) | 115/Q41 | 2h | Detection comparability |
| 2 | D3 full eval, persist per-frame predictions | 115/Q40 | 1h | All published numbers; F22/F22b GPU verification; feeds #5–8 |
| 3 | D4 YOLOv8m→decoder | 115/Q16 | 2–3h | PSR F1 comparability (with #5) |
| 4 | Q43 canonical-order POS baseline (CPU) | 117 | hrs | Gate G4 on the flagship claim |
| 5 | Q17 tau distribution (on D3 artifact) | 117 | hrs | Gate G3; delivers most of E2/Q44 |
| 6 | Q50 TTA + Q1 Soft-NMS on epoch-11 ckpt | 117 | 2–3h | +0.02–0.07 detection, zero training |
| 7 | Q18 PSR per-component thresholds (held-out tuning) | 117 | 1d | F1 0.144→~0.17–0.22, zero training |
| 8 | T4/Q42 act_top1 + T3/Q45 MViTv2 remap | 115/117 | 1h + 1d | Gate G1 (T2 go/no-go) |
| 9 | Fix Anomaly-2 n_present bug + ablation ckpt-dir bug | 112/111 | hrs | Metric bookkeeping integrity |
| 10 | Freeze body-pose branch; Q13 FiLM inspection; Q23 cosine probe (one firing) | 111/117 | hrs | Kendall hygiene + free diagnostics |

## T1 — Before AAIML submission (weeks 2–3, mostly 3060; 5060 Ti after Jul 12–16)

A1-redo + A2–A4 (clean protocol) → B1/Q21 → E1 FPS (measured, both TTA modes) → Q26 discriminative-LR pretrain probe → Q38 pseudo-label branch → Q15 multi-seed (7, 123) → Q11+Q12 pose-loss run → Q36 PSR weighting probe (if Q18 leaves headroom) → Q9+Q35+Q47 activity probe (one resumed 5-epoch run, three knobs) → Q19 smoothing sweep → Q34 offline SWA average.

## T2 — Gated

T2-temporal (gate G1, with Q7+Q8 folded in) · Q5+Q2 OHEM arm (gate G2) · Q10/Q49 GT-fraction probe · Q24 soft-cap probe · Q32 EMA horizon check · Q20 seq-freq (only if G3 says paradigm-limited) · Q30 (only if A-suite blocked) · Q46 (only if Q13 says FiLM is real).

## SKIP for this paper (journal-extension queue, in order of promise)

Q48 MAE pretrain → Q37 synthetic data → Q3 BiFPN → Q27 Swin-T → Q4 head depth → Q6 75-class → Q28 ConvNeXt-S → Q22 GradNorm → Q25 init study → Q29 EfficientNet → Q31 peak factor → Q33 mixup(main) → Q39 active learning.

## The one-sentence summary

Run the cheap comparability and inference-only experiments this week while the main run climbs, gate every expensive commitment (T2, OHEM, PCGrad, cap-removal) on a cheap measurement that prices it first, spend the ablation budget on the efficiency thesis rather than the weakest head, and publish the ego-pose baseline immediately with every caveat stated before a reviewer can discover it.

---

*Cross-references: every verdict above cites its source question/section inline. Evidence base: 111 (full), 112 (§12, Appendices B–D), 113 (§1.1a–1.3), 114 (§5–7), 115 (§7–9 structure), 116 (§1, §3, §8, §10), 117 (all 50 questions + hypotheses). Live-state facts (epoch 12, PID 3432463, idle 3060, epoch-11 metrics) are as of the 2026-07-04 16:57 JST snapshot in 111.*
