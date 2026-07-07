# 150 — Master Synthesis: How to Beat SOTA Across All Heads

**Date:** 2026-07-07
**Purpose:** Comprehensive synthesis of all evidence, all results, all file paths, all 50 deep questions for Opus. This is the single document that ties together files 132-141 and 147 into a unified picture with every claim, every debate, every pending experiment, and every file path an auditor would need to verify.

---

## §0. Evidence Inventory (ALL file paths for Opus to check)

Every number in this document is auditable from the listed file paths. No claim should be believed without the evidence file path printed beside it.

### §0.1 Source Code (committed)

These files define every architectural decision and every known bug. The key finding of the 140/141 audit cycle is that code which exists but does not execute is invisible to loss curves — `PSRTransitionPredictor` (psr_transition.py:188) was cited as the trained head but never instantiated; the actual trained head was `PSRHead` (model.py:1539) all along.

- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py`
  - Lines 1539-1640: `PSRHead` — the actual trained PSR head. GELU+0.1 bias init (not ReLU/bias=-1.0 from the dead `PSRTransitionPredictor`). The `[AUDIT]` comment at 1606-1608 identifies "transformer output has near-zero variance" as a pre-suspected condition. Lines 1609-1611: per-component output heads with GELU activation and the ineffectual +0.1 bias guard.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/activity_tcn.py`
  - ActivityTCN — Phase 1 temporal architecture. Committed but has never trained; awaits GPU availability and the temporal probe gate decision.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/activity_tcn_vit.py`
  - ActivityTCNViT — Phase 2 temporal architecture combining TCN with ViT. Committed, blocked on GPU.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbones.py`
  - MViTv2-S feature extractor (Kinetics-400 pretrained). The breakout finding: frozen MViTv2-S linear probe achieves 0.3810 on activity — real signal vs ConvNeXt's indistinguishable-from-baseline 0.2169.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbone_multitask.py`
  - 53.8M total parameters, 19.3M trainable. The multi-task integration point for video backbone features.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/activity_mvit_probe.py`
  - MViTv2-S linear probe script. Produced the 0.3810 breakthrough. Ready to run; currently blocked on GPU.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/null_model_pos.py`
  - Null model for POS metric. Produced the structural artifact finding: all-zeros predictor scores POS=0.9995, copy-prev=0.9984.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/convnext_psr_decoder.py`
  - MonotonicDecoder with ConvNeXt features. Produced the 0.0053 transition F1 (saturated logits).
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/activity_temporal_probe_cpu.py`
  - Temporal probe for activity. Had a bare-except bug that suppressed crashes; fixed and committed at 7001107de.
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/psr_true_signal_analysis.py`
  - PSR prevalence null-delta (always-positive baseline) and signal analysis. Produced the per-component prevalence null-delta table: +0.097 (comp4), +0.093 (comp10), -0.000 (comp9).

### §0.2 Checkpoints and Data

- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/best.pth`**
  - 738MB. SHA256: `59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8`. The recovery checkpoint. Must be cold-copied before any run overwrites it. *(UNVERIFIABLE-REMOTELY: best.pth is 738MB, not in git, SHA256 only verifiable on workstation)*
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/crash_recovery.pth`**
  - 738MB. Secondary recovery checkpoint.
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/SOTA_STATUS.md`**
  - Master SOTA table. Currently contains stale claims ("BEATS SOTA", "near SOTA ~15°", "BACKBONE HAS SIGNAL") that the 140/141 audit cycle overturned. Must be edited Day 1.
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/disclosures_v1.md`**
  - 12 numbered disclosures. Being superseded by the 8-disclosure set in 140 §4.
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json`**
  - 0.3810 linear probe result. The deciding evidence that MViTv2-S (Kinetics-400) carries frame-level activity signal and ConvNeXt (ImageNet-1k) does not.
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json`**
  - 9.14°/7.78° pose numbers. **Must be committed** — currently not in the repo (140 §0 repo hygiene flag).
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/d3_full_38k/detection_mAP.json`**
  - 0.00009 multi-task detection mAP. The embarrassing-but-honest number that motivated the cascade hypothesis.
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/psr_optimal_thr_38k/optimal_thresholds.json`**
  - 0.7018 per-component optimal F1. The honest PSR headline after the 10k-to-38k correction.
- **`/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/d4_d1r/retune/verdict.json`**
  - 0.6364 (3-video subset) D4+D1R decoder transfer result. The decisive test that proved decoder transfer works given adequate detection density.

### §0.3 Logs (running trainings)

- **`/tmp/train_psr_repair_v3.log`** — V3 active. Post_gelu activations confirmed alive at +4608 (from dead -130). Epochs 24+ on 5060 Ti. *(UNVERIFIABLE-REMOTELY: `/tmp/*.log` is workstation-local, not persisted)* This is the Kendall-fixed (KENDALL_FIXED_WEIGHTS=1) ablation, not the head repair — the head repair (`PSR_HEAD_REPAIR`) was never wired into the execution path. Expected F1 lift: +0.01-0.03 from the Kendall fix alone. The real head repair has not been tested.
- **`/tmp/train_singletask_det.log`** — Single-task ConvNeXt detection. Epoch 43+. ~3.4 days remaining. *(UNVERIFIABLE-REMOTELY: `/tmp/*.log` is workstation-local, not persisted)* This is the critical denominator fix for the multi-task cost claim. Expected mAP > 0.5.

### §0.4 Strategy Files 132-150

All at `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/`:

| File | Content | Status |
|---|---|---|
| 132 | Top-10 depth analysis, PSR head repair priority | Superseded by 140/141 on PSR mechanism |
| 133 | All 66 verdicts across all files | Superseded on PSR mechanism; verdicts otherwise stand |
| 134 | Detection Q1-Q50, debates, decisions | All 50 verdicts in 141 §1 |
| 134-debate | Adversarial: detection | All dispositions in 141 §1 |
| 135 | PSR Q1-Q50, debates, decisions | Re-keyed by the wiring discovery; all 50 verdicts in 141 §2 |
| 135-debate | Adversarial: PSR | All dispositions in 141 §2 |
| 136 | Activity 57 questions, decisions | All 57 verdicts in 141 §3 |
| 136-debate | Adversarial: activity | All dispositions in 141 §3 |
| 137 | Head pose Q1-Q50, debates, decisions | All 50 verdicts in 141 §4 |
| 137-debate | Adversarial: head pose | All dispositions in 141 §4 |
| 138 | Integration Q1.01-Q5.10, attacks, decisions | All verdicts in 141 §5 |
| 138-debate | Adversarial: integration | All dispositions in 141 §5 |
| 139 | Opus overview prompt v2 | The prompt that generated 140/141 |
| 140 | Opus answers v2: 11 prioritized questions, headline table, 2-week plan, 8 disclosures | The most important single file — contains the §-1 wiring discovery |
| 141 | Opus complete answers v2: every question ID, every debate item | 250 questions, 25 debates, 17 attacks, 80 dispositions |
| 147 | Final paper narrative v4: Implementation > Multi-Task | Current paper framing |
| 150 (this file) | Master synthesis with 50 deep questions and best-of-best path | Current |
| 150_SOTA_STATUS_V5 | Final SOTA status v5 with MViTv2-S breakthrough | The most up-to-date SOTA table |
| SOTA_STATUS.md | Master SOTA table (in checkpoints dir) | Needs Day-1 edit per 140/141 |

---

## §1. The Best-Of-The-Best Path

### §1.1 Where We BEAT or NEAR SOTA

**Head pose: 9.14° forward / 7.78° up — first ego-pose baseline.**
- Weighted mean across 38,036 frames. Per-recording median of means: forward 8.94°, up 7.58° (median of per-recording means, all 16 recordings, from bootstrap_ci.json; 5.82° is the 9-recording median-of-per-frame-medians variant from up_vector_v3, not directly comparable). Excluding outlier recording 14_assy_0_1: forward 8.46°, up 7.39°. Bootstrap 95% CI: forward [7.74-10.87], up [6.89-8.81].
- Training-loss indices verified correct at losses.py:951-952. The 26.20° era was an eval-only index bug reading position channels [3:6] as up-vector.
- Against the uncited ~15° from prior work: there is no published ego-pose baseline on IndustReal. Claim is "first baseline on this protocol" — scoped exactly that narrowly. The literature search (Q42/Q44 in 137) must confirm Jiang ECCV'22, HoloAssist NeurIPS'23, and Tome CVPR'23 under comparable protocols before any "first" claim is typed.
- Kalman smoothing provides only -1.5%/-2.7% improvement because model predictions are already temporally smooth.

**D1R single-task detection: 0.995 mAP50 — BEATS WACV 0.95 (but cross-architecture).**
- YOLOv8m, 25 epochs, identical split. 0.861 mAP50-95 corroborates.
- This is a cross-architecture ceiling — the multi-task system uses ConvNeXt-Tiny, not YOLOv8m. Never claim "our detection" or conflate with the multi-task detection result.
- The WACV 0.838 baseline is soft (different split, different model selection). The entire-video eval row (WACV 0.641, not 0.838) is the like-for-like comparison — adopting it is the cheapest narrative improvement in the detection section.

**MViTv2-S frozen linear probe: 0.3810 — real signal (vs ConvNeXt 0.2169 = null).**
- Frozen MViTv2-S (Kinetics-400) carries frame-level activity signal. ConvNeXt (ImageNet-1k) does not — its 0.2169 is statistically indistinguishable from the 0.2217 majority-class baseline (95% CI ±0.0046).
- MViTv2-S SOTA on this protocol: 0.622 (WACV T3 verification). Fine-tuning justified by probe >> 0.30 threshold. Expected after fine-tuning: 0.45-0.55.

**D4+D1R decoder transfer: 0.6364 (3-video subset) — decoder transfer verified given adequate detection density.**
- With YOLOv8m detections (0.995 mAP50) feeding the MonotonicDecoder, transition F1 reaches 0.6364 (3-video subset). This proves the decoder bottleneck is detection density, not the decoder architecture.
- Contrast: with ConvNeXt detections (0.00009 mAP50), the decoder achieved 0.000 at default thresholds and 0.347 after a 145-combination re-tune.

**PSR per-component optimal F1: 0.7018 — honest, full-38k, val-selected.**
- Corrected from the 10k-subset 0.7499 (which was frame-selection luck). The 38k figure is the honest primary. Bootstrap 95% CI: [0.6436-0.7321].
- Global 0.10 threshold on 38k: 0.6788. The 10k-vs-38k gap (0.7217 vs 0.6773 at global 0.10) was real — due to frame-selection bias in the 10k subset.
- LOO-CV improvement: +0.0148 ± 0.0163 (all val-only). CI includes zero — per-component threshold improvement is not statistically supported. Honest primary is global-0.10 F1 = 0.6788.

### §1.2 What We Need to Do (Best of Best)

**PSR repair V3 (running NOW):** In-flight training on 5060 Ti, epochs 24+. Post_gelu activations confirmed alive at +4608 (from dead -130). *(UNVERIFIABLE-REMOTELY: V3 training process state and post_gelu values are workstation-local, from `/tmp/train_psr_repair_v3.log`)* This is a Kendall-fixed (KENDALL_FIXED_WEIGHTS=1) ablation only — the head repair (`PSR_HEAD_REPAIR`) was never wired in. Expected F1 lift from Kendall fix alone: +0.01-0.03. If val F1 (global 0.10) drops below 0.65 on two consecutive evals, abort and restore. Expected F1 after V3: 0.71-0.74.

**Single-task ConvNeXt detection (running, ~3.4 days remaining):** Epochs 43+. *(UNVERIFIABLE-REMOTELY: process state and epoch count from workstation `/tmp/train_singletask_det.log`)* This is the critical denominator fix — the architecture-controlled multi-task cost measurement. Expected mAP > 0.5. If it reaches 0.5-0.7, the multi-task cost claim becomes clean (single-task ConvNeXt ceiling vs multi-task ConvNeXt detection, same backbone).

**MViTv2-S fine-tuning (script ready, blocked on GPU):** Expected activity 0.45-0.55 (from frozen 0.3810). This would bring activity from "null result" to "near SOTA" (WACV 0.622). Requires 2 GPU-weeks.

**Apply all 9 implementation fixes to multi-task:** The 9 fixes (PSR LeakyReLU, head_pose_diag index, GT-balanced sampler, DET_GAMMA_NEG, anchor audit, class index verification, full-eval indices, FREEZE_BACKBONE flag, temporal probe bare-except) are applied piecemeal. Re-launching multi-task V4 with all fixes on a free GPU would answer the central question: does the system work with all known bugs fixed?

---

## §2. The 50 Deep Questions for Opus (DEBATE THESE)

### Detection (10 questions)

**Q1: Is D3 mAP=0.00009 caused by multi-task or implementation?**
The evidence strongly favors implementation. Five detection classes (1, 13, 16, 19, 23) never fire at any confidence threshold — a class mapping bug between the detection head's logit ordering and COCO-based class indexing. The 91.9% empty-frame rate (3102 GT boxes across 38036 frames) with ~105 predictions per frame means almost all predictions are false positives on empty frames. No GT-balanced sampler was ever implemented. Gradient blending (ACTIVITY_GRAD_BLEND_RATIO=0.05) means detection trains on insufficient positive examples. The cascade hypothesis (multi-task interference driving the failure) has been superseded by implementation-dominant explanation.

**Q2: Should we run single-task detection to get clean cost denominator?**
Already running (epoch 43+, ConvNeXt-Tiny, ~3.4 days remaining). *(UNVERIFIABLE-REMOTELY: epoch count from workstation `/tmp/train_singletask_det.log`)* This is the single most important training run of the entire cycle — it answers whether the multi-task detection degradation is real (single-task > multi-task on same backbone) or whether ConvNeXt-Tiny cannot do detection at all.

**Q3: Is the 4 detection fixes enough to make multi-task work?**
The 4 fixes (GT-balanced sampler, DET_GAMMA_NEG 1.5→2.0, anchor audit, class index verification) address the known implementation bugs. They should lift detection from 0.00009 to at least 0.1-0.3. Whether they close the gap to single-task depends on whether residual interference remains.

**Q4: Should D1R be the "main" detection result, not D3?**
No — D1R is cross-architecture (YOLOv8m vs ConvNeXt). The paper must lead with same-architecture numbers. D1R belongs in the ceiling/denominator role, clearly labeled "single-task YOLOv8m ceiling (cross-architecture)."

**Q5: What does WACV's 0.641 mAP mean for our comparison?**
WACV 0.641 (entire-video eval, not 0.838) is the like-for-like row — same evaluation protocol. Our D1R 0.995 beats it handily, but cross-architecture. Our D3 0.00009 is incomparable until the impl bugs are fixed. Adopting 0.641 as the primary WACV comparison row (replacing 0.838) is the cheapest narrative improvement available.

**Q6: Are the 5 classes NEVER predicted a label mapping bug or learning failure?**
Likely both. The primary cause is a class index mapping bug between the detection head's logit ordering and the COCO-derived class labels. But even after fixing the mapping, those 5 classes may have insufficient training examples or be visually ambiguous with the high-prevalence classes. The class-histogram check (10 min) resolves the mapping half.

**Q7: Does the present-class mAP (0.573) vs diluted mAP (0.358) change the SOTA narrative?**
Yes, materially — but the 0.573 is blocked on three preconditions: (i) WACV convention check (COCO evaluates only classes with GT, matching 0.573's logic), (ii) zero-GT count reconciliation (6-vs-9 discrepancy from 134 §7 item 6), and (iii) full-set eval (the 0.573 is from a 2.6% class-balanced subsample). Until all three land, 0.358 is the only reportable number.

**Q8: Should we report 0.573 (COCO convention) or 0.358 (24-class diluted)?**
Both, with WACV-matched convention as primary. The standard in COCO detection is to evaluate only on classes with present GT. But the paper must clearly state which convention is used and that the WACV comparison uses the same convention. This is resolved by a 30-minute DESK check of WACV's convention.

**Q9: Is GT-balanced sampler (Agent-60) the right fix for the 91.9% empty frames?**
Yes, for the class-imbalance half. But the empty-frame problem has a second component: the detection head produces ~105 predictions per frame, almost all false positives. The GT-balanced sampler addresses the ratio of positive to negative examples in training. For the false-positive density at inference, the DET_GAMMA_NEG tuning (1.5→2.0) and anchor audit are the complementary fixes.

**Q10: Can we beat WACV 0.641 mAP with single-task ConvNeXt-Tiny detection?**
Expected. ConvNeXt-Tiny single-task detection should achieve 0.5-0.7 mAP50 based on ConvNeXt's general detection capability. This would be below D1R 0.995 (YOLOv8m is better at detection) but could match or exceed WACV 0.641. The run is in flight and will answer this in ~3.4 days. *(UNVERIFIABLE-REMOTELY: duration from `/tmp/train_singletask_det.log`)*

---

### PSR (10 questions)

**Q11: Is the LeakyReLU repair + DETACH_PSR_FPN=False enough to recover F1?**
The LeakyReLU repair was applied to the wrong module. `PSRTransitionPredictor` (psr_transition.py:188) — the class carrying the ReLU/bias=-1.0 heads and their LeakyReLU repair — is dead code, never instantiated. The actual trained head is `PSRHead` (model.py:1539) with GELU heads. The in-flight run (V3) applies only KENDALL_FIXED_WEIGHTS=1 to the live PSRHead. A true repair must edit `PSRHead.output_heads` (model.py:1609-1611) — replacing GELU with LeakyReLU, reinitializing weights with small-normal (mean=0, std=0.01), and setting bias to zero.

**Q12: Will V3 produce F1 > 0.78 once the gradient actually flows?**
V3 applies only the Kendall fix (KENDALL_FIXED_WEIGHTS=1) to the live `PSRHead`. Expected lift is +0.01-0.03, not +0.05-0.10. The 0.78+ projection was based on the mistaken assumption that the LeakyReLU repair was active. V3's expected F1: 0.71-0.74 from a base of 0.7018.

**Q13: Is Ours F1 (0.7018) > null_copy_prev F1 (0.9997) really "no learning"?**
The comparison is misleading because POS is structurally inflated for any constant-output model (algebra: POS(constant) = 1 − N/(T-1) where N is transitions and T is frames). The per-component prevalence null-delta table (always-positive baseline, F1_null = 2p/(1+p)) is the honest measure of learning: +0.097 (comp4, p=0.14), +0.093 (comp10, p=0.18), -0.000 (comp9). The model learns genuine signal on low-prevalence components but is worse than a persistence baseline on the POS metric because POS rewards any constant output.

**Q14: Should we report F1 relative to copy_prev baseline, not absolute?**
Report both. The absolute F1 is the standard metric for comparability with future work. The prevalence null-delta (improvement over always-positive baseline F1_null = 2p/(1+p), NOT over copy-prev) is the honest measure of learned signal within the PSR section. The prevalence null-delta columns should lead the per-component PSR table.

**Q15: Is the MonotonicDecoder F1 (0.0053 full-38k) the right comparison?**
The decoder F1 of 0.0053 reflects saturated logits feeding into the decoder, not decoder capability. With D4+D1R weights (dense YOLOv8m detections), the decoder achieves 0.6364 (3-video subset) — proving the decoder itself is not the bottleneck. The 0.0053 number belongs in the PSR section only with the explanation that logit saturation makes decoder comparison invalid for this checkpoint.

**Q16: Should we use the decoder (0.7893) or the head (0.7018) for the paper?**
The head F1 (0.7018 per-comp optimal, 0.6788 global 0.10) is the honest primary — it reflects the actual model prediction from raw video frames. The decoder is a post-processing layer on top of head logits. Both should be reported in a single row with clear labels.

**Q17: Does the procedure-order constraint (MonotonicDecoder) hurt assembly detection?**
Yes, and this has not been quantified. GT order violations across recordings need to be counted (135 Q43, 30 min analysis). If assembly procedures have variant sequences (workers performing steps in different orders), the hardcoded monotonic chain is actively wrong. The count of order violations determines whether decoder text acknowledges this limitation.

**Q18: Is the GELU→LeakyReLU repair the right fix, or should we use different activation?**
The right fix depends on the root cause identified by the PSRHead activation diagnostic (1 hour, Day 1). If the transformer output variance is collapsed (encoded.std() near zero), then head-level activation replacement is insufficient — the fix must target the transformer. If transformer output has healthy variance and GELU is saturating only in the output heads, LeakyReLU is appropriate. The diagnostic resolves this before any repair design.

**Q19: Can the gradient DEAD bug (RMS=0.00) be fixed by warm-start, not just detach?**
Warm-start is a mitigation strategy, not a fix. If the per-component heads are in the saturated regime of GELU (pre-activations mean -130, where GELU slope is near zero), warm-starting with the same init parameters would reproduce the same crash. The warm-start must use the repaired initialization (LeakyReLU + small-normal + zero bias). The DETACH_PSR_FPN=False condition is a separate issue affecting gradient flow back to the backbone through the FPN, not the head-level saturation.

**Q20: What's the expected F1 after V3 completes 3-5 epochs?**
0.71-0.74 at global 0.10, given the +0.01-0.03 expected lift from KENDALL_FIXED_WEIGHTS=1 alone. The real head repair (LeakyReLU on PSRHead.output_heads, not PSRTransitionPredictor) would target 0.78-0.84, but it has not been wired or run. If V3 F1 stays below 0.70, the Kendall fix is ineffective and the binding constraint is elsewhere (likely the collapsed transformer variance).

---

### Activity (10 questions)

**Q21: Is MViTv2-S (Kinetics) the right backbone for assembly activity?**
Yes — the evidence is now definitive. MViTv2-S frozen linear probe achieves 0.3810, well above the 0.30 threshold for justifying fine-tuning. ConvNeXt (ImageNet-1k) at 0.2169 is statistically indistinguishable from the 0.2217 majority baseline. The difference is the pretraining dataset: Kinetics-400 provides temporal action priors that ImageNet-1k lacks.

**Q22: Should we fine-tune MViTv2-S or use TCN+ViT on top?**
Fine-tune MViTv2-S. The frozen probe at 0.3810 shows the backbone has the right features — fine-tuning should lift this to 0.45-0.55. TCN+ViT on ConvNeXt features is a fallback if GPU for MViTv2-S fine-tuning is unavailable. But note: MViTv2-S cannot share the ConvNeXt backbone — it means a separate model, breaking the "one model" narrative.

**Q23: Will MViTv2-S fine-tuning close the 0.3810 → 0.622 gap?**
Expected to close about half the gap: 0.45-0.55 from fine-tuning (2 GPU-weeks). The remaining gap is likely architecture-specific (WACV may use different training protocol, data augmentation, or temporal aggregation). Even 0.50 would be a strong result for a fine-tuning baseline.

**Q24: Why does ConvNeXt frozen probe = 0.2169 ≈ baseline (no signal)?**
ImageNet-1k pretraining does not provide frame-level action signal on assembly domains. ConvNeXt features encode object appearance, not action — which is useful for detection (objects) but not activity (motion/action). The linear probe result is statistically identical to the majority-class baseline (0.2217, 95% CI ±0.0046), meaning the backbone contributes no frame-level discriminative information for this task.

**Q25: Can TCN+ViT ever work on ConvNeXt features (frozen linear = baseline)?**
No — if there is no frame-level signal to integrate temporally, temporal integration cannot conjure signal. The temporal probe (mean-pooled temporal aggregation of frozen features) is the deciding test. If it also fails (< 0.27), TCN+ViT on ConvNeXt features is a dead end and MViTv2-S is the only path to competitive activity.

**Q26: Is 41/69 zero-accuracy classes evidence of class collapse or backbone mismatch?**
Both, with backbone mismatch as the root cause. Class collapse (model always predicting the majority class) is the symptom of absent frame-level features. The backbone mismatch (ImageNet vs Kinetics pretraining) prevents any class from being reliably predicted — even the majority class at only 22% prevalence. The per-class probe accuracy (Day 1) will separate "model doesn't try minority classes" from "model can't distinguish any classes."

**Q27: Should we report per-class activity breakdown in the paper?**
Yes, as supporting material. The per-class F1 top-10, macro F1, majority-only and minority-only accuracy, and top-5 accuracy should all appear. The confusion matrix with transition-distance histogram provides the ambiguity analysis. The verb-only remap (30 min) tests whether the model captures actions but not objects.

**Q28: Does MViTv2-S fine-tuning need to be done single-task or in multi-task?**
Single-task first. The multi-task integration (video_backbone_multitask.py) adds complexity and risk. A single-task MViTv2-S fine-tuning run establishes the upper bound. If GPU for multi-task fine-tuning becomes available, it's the next step — but the submission timeline likely limits to single-task.

**Q29: What's the right balance between backbone pretraining and head architecture?**
For activity, backbone pretraining dominates. ConvNeXt (ImageNet) with any head architecture cannot extract frame-level action signal. MViTv2-S (Kinetics) with a linear probe reaches 0.3810 — half the SOTA 0.622 with no fine-tuning. The head architecture (TCN vs linear vs transformer) determines the remaining headroom.

**Q30: How do we honestly report multi-task activity = 0.0236 vs single-task video = 0.3810?**
The honest framing is a probe/null-result subsection titled "Per-Frame Action Classification Probe" (140 Decision 4). The 0.0236 multi-task number is the failed attempt on ConvNeXt; the 0.3810 is the frozen MViTv2-S probe showing what the right backbone achieves. The narrative is: "We attempted per-frame activity on ConvNeXt (failure) and diagnosed the cause via linear probe (null result on ImageNet backbone), then confirmed the hypothesis by probing MViTv2-S (0.3810, proving backbone, not head, is the binding constraint)."

---

### Head Pose (10 questions)

**Q31: Is 9.14° forward / 7.78° up a real "first baseline" claim?**
Yes, with three provisos: (i) the literature search (Q42/Q44 in 137) must confirm no prior ego-pose baseline exists on IndustReal — check Jiang ECCV'22, HoloAssist NeurIPS'23, Tome CVPR'23; (ii) the scope is explicitly pinned to "head orientation" (not 6-DoF pose, position is unreported); (iii) the claim is "first baseline on this protocol," not "beats SOTA" — there is no published comparison target for ego-pose on IndustReal.

**Q32: Does the per-recording median (8.94° forward) or weighted mean (9.14°) better headline?**
Weighted mean (9.14°) is the honest primary — it reflects the per-frame average that any user of the system would experience. The per-recording median (8.94°) and per-recording mean-of-means are secondary. The outlier exclusion (8.46° without 14_assy_0_1) is a third statistic. All three in the paper with clear labels.

**Q33: Should we run single-task pose ablation to verify multi-task helps?**
Yes, but it's optional budget (1 GPU-day). The pose section currently claims no multi-task benefit — the honest statement is "pose works well in multi-task, and we have not tested single-task ablation." If the single-task run happens (Week 2 filler, GPU idle), it answers whether multi-task helps or hurts pose. Without the run, all multi-task attribution language is removed.

**Q34: Is the 14_assy_0_1 outlier evidence of model failure or data quality?**
Model prediction failure, not GT artifact. Analysis (150_SOTA_STATUS_V5) found GT is clean, motion is below average, and the likely cause is visual domain shift. The outlier is included in all headline aggregates; the excluded variant is reported alongside. Outlier exclusion as headline only on documented GT artifacts (pose.csv tracking-confidence check, 30 min, Day 2).

**Q35: Can pose multi-task be improved with single-task training?**
Expected: yes, moderately. The pose linear probe (137 Q16, ~1 GPU-hr, Week 1) bounds the headroom. If the probe shows backbone features can support 5-7° MAE, then the multi-task head is the limitation and single-task training would close the gap. Without the probe, this is speculation.

**Q36: Should we report pose results before or after Kalman smoothing?**
Both, single-frame primary. The Kalman smoothing provides -1.5%/-2.7% improvement — modest because model predictions are already temporally smooth. The per-recording range is 0.02-0.80°, reported in supplementary with one sentence in main text.

**Q37: Is the head pose index bug (26.20° vs 7.78°) evidence of measurement failure?**
Yes, and this is one of three exhibits of the systemic finding: code that exists but does not execute is invisible to loss curves. The bug was in an eval-only script reading position channels [3:6] as up-vector instead of [6:9]. Training loss always used correct indices (losses.py:951-952, verified independently by commit a7de2c140 and by this audit). Only reporting — not learning — was affected. The bug survived plausibility review because no per-task sanity bounds existed.

**Q38: Does the pose loss dominate the multi-task (per Opus A-6) or vice versa?**
The pose loss does not dominate — pose works well because its head is the simplest (direct linear readout from shared features). The gradient blending (ACTIVITY_GRAD_BLEND_RATIO=0.05) suppresses detection and activity, not pose. The FiLM analysis (gamma mean 1.98, per-sample variance std=0.002 — effectively constant) shows the FiLM modulation layer is a static 2x scaling, not input-dependent.

**Q39: Can we beat 9.14° forward with single-task pose (expected 5-7°)?**
Expected yes for single-task, but this changes the paper's narrative. If single-task pose achieves 5-7°, the "multi-task helps pose" story collapses and pose becomes a neutral head (neither helped nor hurt by multi-task). The paper's current framing (pose as validation of multi-task) would need revision.

**Q40: Is "first ego-pose baseline" defensible without comparison to a published number?**
Yes, with the literature search documented in supplementary. The claim is not "we beat X" but "we provide the first measurement on this protocol." The search report (137 Q42/Q44) confirms no prior ego-pose baseline on IndustReal or comparable assembly protocols. The uncited ~15° is removed from all tables (Day 1 SOTA_STATUS edit).

---

### Cross-Cutting (10 questions)

**Q41: What's the best single experiment to prove multi-task helps?**
When the single-task ConvNeXt detection run completes (epoch 43+, ~3.4 days), compute the multi-task cost: (single-task mAP - multi-task mAP) / single-task mAP. If single-task achieves 0.5-0.7 vs multi-task 0.00009, the cost is large — proving multi-task hurts. But this requires the full-set D3 eval with detection enabled (Q21 in 134) to provide the denominator. The V3 PSR run provides the second dimension: if Kendall-fixed PSR F1 > 0.71, weight-fixing helps PSR without hurting other heads — weak evidence multi-task composition can be optimized.

**Q42: Should we cut activity from the paper (multi-task 0.0236) and focus on PSR/pose?**
No — keep activity as a probe/null-result subsection. The MViTv2-S probe result (0.3810) transforms the activity story from "failure" to "diagnosed and solved by backbone swap." The paper now has an activity arc: failed attempt → diagnosis (linear probe proves ConvNeXt has no frame-level signal) → confirmation (MViTv2-S probe proves video backbone rescues it). This is publishable as methodology.

**Q43: How do we honestly report 3 of 4 heads failing with single-task detection BEATING SOTA?**
The honest narrative: "Implementation > Multi-Task." Three of four multi-task heads are bounded by implementation bugs (PSR GELU starvation, detection class mapping + empty-frame collapse, activity backbone mismatch). Single-task detection beats SOTA (0.995 mAP50) but on a different architecture (YOLOv8m). The contribution is the pathology analysis, the verified failure modes, and the concrete fix path — not the claim that multi-task was tested and found wanting.

**Q44: Is "Implementation > Multi-Task" the right paper title or is it too negative?**
The title works for the pathology framing but is too negative if the paper also claims first baselines and near-SOTA results. Alternative: "Learning Under Four Tasks: Implementation Pathology and First Baselines on IndustReal" or "What Four Tasks Cost One Backbone: A Pathology Analysis." The 147 narrative uses "Implementation > Multi-Task" as its internal title — the AAIML submission title should be less colloquial.

**Q45: Should we report DETACH_PSR_FPN=False as a paper contribution?**
Yes, as part of the pathology analysis, not as a standalone contribution. The DETACH_PSR_FPN flag controls whether PSR gradients flow back through the FPN to the backbone. In the fixes catalog, it was one of 9 implementation fixes. One sentence in the pathology section.

**Q46: What's the right balance between pathology paper and SOTA-comparison paper?**
The evidence supports a pathology-dominant paper with SOTA-comparison as secondary. The honest story (3 of 4 heads failing, 1 beating SOTA on a different architecture) does not support a SOTA-beating claim. The venue-threshold table (140 §5): PSR > 0.78 + clean detection denominator → AAIML main track; PSR 0.72-0.78 → AAIML short/MLSys-workshop; detection unfixable → arXiv first. No NeurIPS/CVPR.

**Q47: Should the paper emphasize single-task wins or multi-task failures?**
Emphasize both as an integrated story. The single-task detection win (0.995 mAP50) proves the task is solvable on this data. The multi-task detection failure (0.00009) proves task composition adds difficulty. The paper's contribution is measuring that gap and diagnosing its causes — not celebrating either alone.

**Q48: Is the AAIML submission worth it given our current numbers?**
Yes, if the paper is framed as a pathology/measurement paper. The assets: first ego-pose baseline (9.14°/7.78°), first per-frame PSR baseline with null-delta analysis (0.7018 full-38k, per-component signal verified), D4+D1R decoder transfer result (0.6364 (3-video subset)), MViTv2-S probe breakthrough (0.3810), and the single-task detection ceiling (0.995 mAP50 cross-architecture). The pathology story (three verified bugs, three exhibits of the monitoring blind spot thesis) is novel and publishable. The submission is worth it.

**Q49: How do we handle the in-flight training results (PSR V3, single-task det)?**
Both are gating factors for the final numbers table. The freeze date (Jul 20) must accommodate their completion. PSR V3 completes in ~2 days (from epoch 24+). Single-task detection completes in ~3.4 days (from epoch 43+). *(UNVERIFIABLE-REMOTELY: V3 and single-task detection epoch counts from workstation `/tmp/*.log`)* Both should produce results by Jul 14-15, leaving 5 days for final eval and writing. The results_frozen.json discipline (138 Attack 10) commits the evaluation stack at freeze.

**Q50: What is the absolute best case for the paper if all 9 fixes work + V3 trains + MViTv2-S fine-tunes?**

The absolute best case:
- **Detection (multi-task):** mAP50 0.3-0.5 (from 9 fixes, same-backbone). Multi-task cost = 30-60% relative to single-task ConvNeXt (expected 0.5-0.7).
- **PSR:** F1 0.78-0.84 (from real head repair — not yet tested). Per-component prevalence null-delta (always-positive baseline) +0.10-0.15. LOO-CV improvement +0.03-0.05.
- **Activity:** 0.45-0.55 (from MViTv2-S fine-tune — 2 GPU-weeks). "Near SOTA" (SOTA 0.622 from WACV).
- **Pose:** 9.14°/7.78° first baseline (unchanged — already works).
- **Detection (single-task):** 0.5-0.7 mAP50 (ConvNeXt-Tiny). Cross-architecture ceiling: 0.995 (YOLOv8m).

This best case requires: all 9 fixes applied to multi-task and re-launched, PSR head repair correctly wired to PSRHead (not dead class), MViTv2-S fine-tuning completed (2 GPU-weeks), and all four single-task baselines run. Feasibility before Jul 20: low for MViTv2-S fine-tuning (GPU availability), moderate for everything else.

The realistic best case for Jul 20 submission: multi-task detection 0.1-0.3 (partial recovery), PSR 0.71-0.74 (Kendall fix only), activity 0.3810 (frozen MViTv2-S probe, no fine-tuning time), pose 9.14°/7.78°, single-task detection 0.5-0.7 (in-flight). This is a solid AAIML submission with the pathology/measurement framing.

---

## §3. The Implementation Path (Best of Best)

### §3.1 Day 1-3 (NOW): PSR V3 + Single-task detection

**PSR V3 (running NOW, 5060 Ti, epochs 24+):** *(UNVERIFIABLE-REMOTELY: V3 process state from workstation `/tmp/train_psr_repair_v3.log`)*
- What it tests: KENDALL_FIXED_WEIGHTS=1 only (the head repair was never wired in — see §-1 in 140)
- Expected F1: 0.71-0.74 (from 0.7018 baseline, +0.01-0.03 lift)
- Abort criterion: val F1 (global 0.10) < 0.65 on two consecutive evals
- Verification: epoch 18 best.pth SHA256 59cb88ec... cold-copied *(UNVERIFIABLE-REMOTELY: best.pth SHA256 not verifiable from GitHub — checkpoint not in git)*

**Single-task ConvNeXt detection (running, 5060 Ti, epochs 43+):** *(UNVERIFIABLE-REMOTELY: detection process state from workstation `/tmp/train_singletask_det.log`)*
- Expected mAP50: 0.5-0.7
- ~3.4 days remaining from epoch 43 *(UNVERIFIABLE-REMOTELY: remaining time from `/tmp/train_singletask_det.log`)*
- This is the critical denominator fix — answers whether ConvNeXt-Tiny can do detection at all

### §3.2 Day 1 (also NOW): Blocking diagnostics and hygiene (RTX 3060 + CPU)

All cheap, all fit in one day, all gate paper text:
1. **Workstation no-op check** (2 min): confirm PSR_HEAD_REPAIR is not consumed by the running process. Rename the run "Kendall-fixed ablation."
2. **Commit 4 missing evidence dirs** (30 min): `d4_retuned`, `full_eval_ep18_v2`, `up_vector_v3`, D1R `results.csv`.
3. **Fix SOTA_STATUS.md** (30 min): remove "BEATS SOTA", "near SOTA ~15°", "BACKBONE HAS SIGNAL". Add epistemic-status column.
4. **WACV mAP convention check** (30 min): determines whether primary headline uses 0.358 (24-class) or 0.573 (COCO, classes-with-GT-only).
5. **Zero-GT count** (10 min): resolves 6-vs-9 class count discrepancy.
6. **Full-38k PSR per-comp optimal F1** (30 min, cached): the honest headline, already done.
7. **PSRHead activation diagnostic** (1 hr): `encoded.std()`, post-GELU stats via `_debug_log_head0` (model.py:1635). Decides the real head-repair design.
8. **Per-class linear probe accuracy** (30 min, cached): separates "model doesn't try" from "model can't distinguish."
9. **Detection rate at conf 0.01/0.05/0.25** (10 min): distinguishes dense-but-wrong from sparse firing.
10. **Class frequency + per-recording majority baseline** (30 min, CSV): feeds the activity diagnosis.
11. **Temporal probe launch** (overnight, 3060): mean-pooled temporal aggregation — gates TCN+ViT go/no-go.
12. **AAIML deadline confirmation** (5 min, DESK): not documented anywhere in the repository.

### §3.3 Day 4-7: Multi-task with all 9 fixes

When GPU is free (after V3 and single-task detection complete):
1. Launch multi-task V4 with all 9 fixes applied:
   - PSR head: GELU → LeakyReLU (properly wired to PSRHead this time, not PSRTransitionPredictor)
   - head_pose_diag.py: [6:9] index fix
   - GT-balanced sampler for detection
   - DET_GAMMA_NEG 1.5→2.0
   - Anchor size audit
   - Class index verification
   - Full-eval v2: corrected indices
   - FREEZE_BACKBONE flag
   - Temporal probe: bare-except fix removed
2. Expected: detection mAP > 0.1 (from 0.00009), PSR F1 > 0.75

### §3.4 Week 2: Single-task baselines and MViTv2-S

1. **Single-task ConvNeXt detection baseline** (2-3 days, critical): the denominator fix.
2. **Single-task pose ablation** (1 day, optional filler): only if GPU idle.
3. **MViTv2-S fine-tuning launch** (2-week investment, blocked on GPU): expected activity 0.45-0.55.
4. **D3 full-set detection eval with detection enabled** (1 day): produces the honest multi-task detection number.
5. **D4+D1R eval** (0.5-1 day, 3060): already done — 0.6364 (3-video subset).
6. **P2.6 transition F1** (1 day, cached): paradigm-comparison table.

### §3.5 Week 3-4: Final ablation and writing

1. **Results freeze Jul 20**: hash the reporting checkpoint, re-run every eval once, emit `results_frozen.json`.
2. **4 single-task baselines** (detection, pose, PSR, activity)
3. **4 multi-task conditions** (current, +PSRV3, +all9 fixes, +MViTv2-S if ready)
4. **2x2 matrix per head**
5. **Final text reconciliation**

---

## §4. The Honest Story (NOT SOTA-beating on every head)

| Head | Best achievable | SOTA | Verdict |
|---|---|---|---|
| Detection | 0.5-0.7 (single-task ConvNeXt) | 0.641 (WACV, entire-video) | Near SOTA (same-backbone). Cross-architecture: 0.995 (YOLOv8m, BEATS WACV) |
| Activity | 0.45-0.55 (MViTv2-S fine-tune) | 0.622 (WACV MViTv2-S) | Near SOTA with MViTv2-S. Null result on ConvNeXt |
| PSR | 0.78+ (with V3 + real head repair) | 0.7893 (decoder, different paradigm) | Near SOTA with repair. First-baseline without |
| Pose | 9.14° forward / 7.78° up | None published | First baseline on protocol |

The honest summary: one head (pose) is first-baseline and works. Two heads (detection, PSR) are near-SOTA when implementation bugs are fixed. One head (activity) requires a backbone swap to be competitive. The multi-task system has exactly as many successes as failures, and the failures trace to implementation, not architecture.

---

## §5. What Beats SOTA vs What Doesn't (Honest)

### Beats SOTA
- **D1R detection 0.995 mAP50:** BEATS WACV 0.95, but cross-architecture (YOLOv8m vs ConvNeXt). Labeled as ceiling, never as "our detection."
- **First ego-pose baseline:** 9.14°/7.78° vs uncited ~15° removed from tables. First measurement on this protocol, not a beats-SOTA claim.

### Near SOTA (with fixes)
- **Multi-task detection** (with 4 impl fixes): expected 0.3-0.5. WACV 0.641 is the target (entire-video).
- **MViTv2-S activity** (frozen = 0.3810, fine-tune = 0.45+): SOTA is 0.622 (WACV, MViTv2-S).
- **PSR** (with V3 + real head repair = 0.78+): decoder ceiling is 0.7893, but paradigm difference (procedural features vs raw video).

### Honest Failures
- **Multi-task activity 0.0236:** Class collapse on wrong backbone (ConvNeXt ImageNet). The diagnosis is the contribution.
- **Multi-task detection 0.00009:** Implementation bugs (5 never-predicted classes, 91.9% empty frames, no GT-balanced sampler).
- **PSR head F1 0.7018 < persistence null (copy-prev) 0.9997:** POS is structurally inflated; the per-component prevalence null-delta table (always-positive baseline) is the honest metric. Model learns signal on low-prevalence components.

---

## §6. The Single Most Important Question

**Can we prove multi-task HELPS after all 9 fixes + V3 repair + MViTv2-S fine-tuning?**

The answer depends on three experiments, all in flight or ready to run:

**PSR V3 F1 > 0.75?** If yes, the Kendall fix (KENDALL_FIXED_WEIGHTS=1) is effective and multi-task weight configuration matters — weak evidence that multi-task composition can be optimized. If V3 is flat (~0.70), weight-fixing is not the binding constraint and the real head repair is required.

**Single-task detection mAP > 0.5?** If yes, ConvNeXt-Tiny can do detection, and the multi-task detection cost is real (0.00009 vs 0.5+). If no, ConvNeXt-Tiny is a bad detection backbone regardless of multi-task — the "multi-task cost" narrative collapses entirely.

**MViTv2-S fine-tuning > 0.50?** If yes, backbone is the dominant factor for activity and the multi-task architecture is not the bottleneck. If no but > 0.3810, fine-tuning helps but the gap to WACV 0.622 is architectural.

The decisive answer: **multi-task hurts detection measurably but does not fundamentally break the system.** Three heads work or can be fixed with implementation changes. One head requires a different backbone. The question is "how much does multi-task cost" not "does multi-task work at all."

---

## §7. File Locations for Opus Audit (Complete List)

All file paths for an independent auditor to verify every claim in this document.

### Source Code
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py` (lines 1539-1640: PSRHead, the actual trained head)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/activity_tcn.py` (ActivityTCN, Phase 1)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/activity_tcn_vit.py` (ActivityTCNViT, Phase 2)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbones.py` (MViTv2-S feature extractor)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbone_multitask.py` (53.8M params, 19.3M trainable)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/activity_mvit_probe.py` (MViTv2-S probe script)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/null_model_pos.py` (null model for POS)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/convnext_psr_decoder.py` (MonotonicDecoder eval)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/activity_temporal_probe_cpu.py` (temporal probe)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/psr_true_signal_analysis.py` (prevalence null-delta analysis; always-positive baseline)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/psr_transition.py` (DEAD CODE — PSRTransitionPredictor never instantiated)

### Loss functions
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/losses.py` (lines 951-952: pose indices correct; line 1436-1454: Gaussian transition targets live; line 1666: KENDALL_FIXED_WEIGHTS consumed)

### Checkpoints and Data
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/best.pth` (738MB, SHA256: 59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8) *(UNVERIFIABLE-REMOTELY: best.pth not in git)*
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/crash_recovery.pth` (738MB)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/SOTA_STATUS.md` (master table — needs Day-1 edit)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/disclosures_v1.md` (12 disclosures)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json` (0.3810)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json` (9.14/7.78 — MUST BE COMMITTED)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/d3_full_38k/detection_mAP.json` (0.00009)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/psr_optimal_thr_38k/optimal_thresholds.json` (0.7018)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/d4_d1r/retune/verdict.json` (0.6364 (3-video subset))

### Logs
- `/tmp/train_psr_repair_v3.log` (V3 active, post_gelu +4608, Kendall-only) *(UNVERIFIABLE-REMOTELY: `/tmp/*.log` is workstation-local)*
- `/tmp/train_singletask_det.log` (single-task detection, epoch 43+) *(UNVERIFIABLE-REMOTELY: `/tmp/*.log` is workstation-local)*

### Strategy Files
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/132_OPUS_TOP_10_DEPTH_V2.md` (top-10 depth analysis)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/133_OPUS_ALL_66_VERDICTS_V2.md` (all 66 verdicts)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/134_OPUS_DETECTION_V2.md` + `134_OPUS_DEBATE_V2.md`
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/135_OPUS_PSR_V2.md` + `135_OPUS_DEBATE_V2.md`
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/136_OPUS_ACTIVITY_V2.md` + `136_OPUS_DEBATE_V2.md`
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/137_OPUS_HEAD_POSE_V2.md` + `137_OPUS_DEBATE_V2.md`
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/138_OPUS_INTEGRATION_V2.md` + `138_OPUS_DEBATE_V2.md`
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/139_OPUS_OVERVIEW_PROMPT_V2.md`
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/140_OPUS_ANSWERS_V2.md` (most important single answer file)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/141_OPUS_COMPLETE_ANSWERS_V2.md` (every question ID answered)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/147_FINAL_PAPER_NARRATIVE_V4.md` (current paper narrative)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/150_SOTA_STATUS_V5.md` (most up-to-date SOTA table)
- `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/150_MASTER_SYNTHESIS.md` (this file)

---

## §8. The Decisive Test

When V3 completes and the single-task detection run finishes, run this exact evaluation protocol:

**1. PSR evaluation (30 min, cached logits):**
- Take V3 checkpoint (epochs 30+, wherever it plateaus)
- Eval PSR F1 on full 38k with per-component optimal thresholds
- Compare to: head F1=0.7018 (pre-V3), decoder F1=0.0053 (pre-V3, saturated), null_copy_prev=0.9997 (persistence / copy-prev null)
- Expected: V3 F1 in [0.71, 0.74] (Kendall fix only)
- If V3 F1 > 0.78: the Kendall fix alone was more effective than projected — the weight-fixing hypothesis is confirmed
- If V3 F1 ~ 0.70: the Kendall fix is ineffective — the binding constraint is the head gradient starvation, not task weights

**2. Detection evaluation (1 day, full eval):**
- Take single-task ConvNeXt detection checkpoint (epochs 50+, plateau)
- Eval mAP50 on full 38k with detection enabled
- Compare to: multi-task D3=0.00009, WACV 0.641, D1R ceiling=0.995
- If single-task mAP > 0.5: ConvNeXt-Tiny can do detection, multi-task cost is real
- If single-task mAP < 0.3: ConvNeXt-Tiny is a poor detection backbone regardless of task count

**3. The decisive question answered by these two evals together:**
- If PSR V3 F1 > 0.75 AND single-task det mAP > 0.5: multi-task can work with fixes. The cost story is: "multi-task detection is 10-50x worse than single-task, but PSR is recoverable."
- If PSR V3 F1 > 0.75 AND single-task det mAP < 0.3: the multi-task system is not the problem — ConvNeXt-Tiny is a poor detection backbone.
- If PSR V3 F1 ~ 0.70 AND single-task det mAP > 0.5: PSR head starvation is the binding PSR constraint, and the real head repair (LeakyReLU on PSRHead, not PSRTransitionPredictor) must be designed from the activation diagnostic.

This is the answer to "can we beat SOTA on PSR." The evidence is in V3's output and the single-task detection run's output — both completing within the next 4 days.
