# 107 — Twenty Deep Questions for Opus: Post-Fix Reality Check

**Date:** 2026-07-03
**Context:** After epoch 5 validation (combined=0.241, RF4 gate passed). All 26 fixes active.

---

## Section 1: PSR & MonotonicDecoder (Q1-Q4)

**Q1 — MonotonicDecoder dimension crash:** The epoch 5 val shows `[PSR METRICS] Failed: only 0-dimensional arrays can be converted to Python scalars`. The MonotonicDecoder expects (N, 11) binary predictions but receives a dimension-squeezed variant. Where exactly is the shape mismatch — in the decoder's input collation or in the frame-level prediction aggregation? Is this a one-line fix (reshape) or a deeper protocol issue?

**Q2 — PSR detach_psr_fpn timing:** PSR binary accuracy rose from 0.291→0.554 with detach_psr_fpn=True (no backbone gradient). This suggests the head CAN learn from FPN features alone. At what epoch should we flip detach_psr_fpn=False to let PSR start shaping backbone representations? Is there a risk of PSR corrupting detection features at that point given the F1 snapshot-restore guard?

**Q3 — PSR unique patterns stuck at 5/2048:** Despite binary accuracy improving, unique state vectors remain at 5 (up from 4). Ground truth shows 20+ patterns per 1000 frames. Is the PSR head converging to a constant state vector with minor variations? Would increasing sequence length from 8→16 help, or is the issue that fill-forward labels create a degenerate learning signal where the optimal strategy is to predict the current state and never transition?

**Q4 — PSR per-component prevalence and loss weighting:** Component prevalence ranges 0.19-1.0. The PSR focal loss uses alpha=0.25 with gamma_pos=0, gamma_neg=2. For a component with prevalence 0.19, the effective positive weight is ~0.05 — barely learning on rare transitions. Should we use per-component alpha (inverse prevalence) instead of the shared 0.25? Would this accelerate unique pattern discovery?

## Section 2: Detection & Focal Loss (Q5-Q8)

**Q5 — dp_scores mean=0.333 at epoch 5:** The mean detection score rose from 0.036 (bias init) to 0.333. Is this the FOCAL_ALPHA 0.25→0.50 fix (F8) taking effect, or the F1 backbone gradient restoration? How much further should we expect scores to separate? Is 0.50-0.60 mean achievable by epoch 20, or is there a ceiling with asymmetric gamma?

**Q6 — mAP50_pc=0.339 vs mAP50=0.212 (0.127 gap):** The 9 zero-GT background channels dilute the COCO-24 metric. For the paper, should we report mAP50_pc as the primary metric and mAP50 as a secondary? Or is there a principled way to exclude zero-GT channels from the denominator? How do other IndustReal papers handle this?

**Q7 — 15/24 classes detected at both epoch 2 and 5:** The number of detected classes hasn't increased. Is the model specializing on the 15 easiest classes and ignoring the other 9? How do we push it to explore the remaining classes — per-class alpha, label smoothing, or more training?

**Q8 — FOCAL_ALPHA=0.50 with gamma_pos=0 vs gamma_symmetric=2:** The asymmetric gamma formulation was designed for extreme class imbalance. At epoch 5 with dp_scores mean=0.333 and max=0.998, the detection head is now producing confident predictions. Should we switch to symmetric gamma (gamma_pos=gamma_neg=2.0) at RF8+ to refine the confident predictions rather than continuing to suppress negatives?

## Section 3: Activity Recovery & Architecture (Q9-Q12)

**Q9 — Activity recovered from 5/69 to 48/69 classes:** Which 21 classes remain unpredicted? Are they the rarest tail classes (<50 frames in training)? If so, should we accept that macro-F1 is capped at ~0.85 of maximum (missing the tail), or should we use class-weighted CE to force exploration?

**Q10 — ACTIVITY_HEAD_SIMPLE=True vs temporal path:** The simple MLP is performing well (0.097 macro-F1). At what epoch should we switch to ACTIVITY_HEAD_SIMPLE=False to enable the TCN+ViT temporal path? Is there a risk that the temporal path will destroy the progress the MLP has made?

**Q11 — Activity loss oscillating:** Activity loss ranges 0.33-1.94 across epoch 5 with no clear trend. Is this normal for a recovering classifier, or is there residual instability from the F18 double-ramp fix? Should we expect monotonic improvement from epoch 6+?

**Q12 — Top-5 accuracy=0.381 vs macro-F1=0.097:** The 4x gap suggests the model gets the activity class in its top-5 often but rarely at rank 1. Is this a calibration issue (logit temperatures too flat) or a genuine ambiguity in the 69 verb-grouped classes (multiple verbs for the same action)?

## Section 4: Multi-Task Balancing & Kendall (Q13-Q16)

**Q13 — Kendall steady state: lv_det=0.125, lv_act=0.04, lv_psr=-0.079, eff_pose=0.125:** Detection and pose have equal effective weight (~0.88 precision each). Activity is getting less weight (0.96 precision — lighter because it's harder). PSR is slowly gaining weight (precision 1.08, becoming harder). Is this a healthy equilibrium for 4-task training? Should we intervene if lv_act drops below 0 (meaning activity is considered "easy")?

**Q14 — combined_v2 not yet logged (F20):** The F20 fix adds degree-normalized pose term to combined_v2. At epoch 5 pose MAE=8.92°, the pose term changes from 0.15*(1/(1+0.1006))=0.136 to 0.15*(1/(1+8.92))=0.015 — a 9x reduction. This will make combined_v2 much lower than combined for detection-dominated runs. For RF4→RF10 gate decisions, should we use combined_v2 as the primary gate metric or keep the old combined?

**Q15 — KENDALL_HP_PREC_CAP: is it still needed?** The cap pins lv_pose at -1.000 (fossil) while effective precision tracks lv_det. Now that pose is well-converged (8.92°), is the cap doing anything useful? If removed, would lv_pose naturally converge to ~lv_det anyway?

**Q16 — Combined metric at 0.241 vs best_metric=0.0:** The gate passed at epoch 5 with patience=0. If the next val shows any improvement, patience resets. What's the expected improvement rate at peak LR (epoch 12)? Should we expect combined to increase by 0.03-0.05 per validation, or are we in diminishing returns territory?

## Section 5: Infrastructure & Paper Strategy (Q17-Q20)

**Q17 — Ablation A timeline:** The single-task presets exist (F16) and take ~2 days each on the 5060 Ti. We have an idle RTX 3060 that could run them in parallel. Should we start the ablation suite now, or wait until epoch 12 (peak LR) for stable baseline numbers?

**Q18 — MonotonicDecoder fix priority:** The PSR eval bug makes all transition metrics invisible. Is this a one-line reshape fix or a deeper issue? Should we prioritize fixing it before the epoch 8 validation?

**Q19 — Xorg GPU assignment:** Doc 102 identified that Xorg claiming both GPUs may trigger the cudaErrorLaunchTimeout. Current run has 0 errors at 3h+, suggesting the F18 fix (activity gradient stabilization) resolved the immediate hang. Should we still reconfigure Xorg to only use the RTX 3060, or is this no longer urgent?

**Q20 — Paper venue confirmed:** With combined=0.241 at epoch 5, pose at 8.92°, and activity recovering to 48/69 classes, is AAIML 2027 main track still the right target? Or should we aim higher (WACV 2027 workshop, ICRA workshop) given that head pose alone is a publishable contribution?
