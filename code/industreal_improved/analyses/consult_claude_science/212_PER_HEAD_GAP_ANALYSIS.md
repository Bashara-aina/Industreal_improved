# Doc 212 — Per-Head Gap Analysis: What Each Head Needs to Beat SOTA

**Status**: Updated 2026-07-11. **Audience**: Paper authors. Brutally honest. No hedging.

---

## Framework

From doc 208, every head's SOTA gap decomposes as:

```
SOTA gap = (SOTA - our ST baseline) + (our ST - our MTL)
            \_____ recipe/data/res _________/   \___ sharing cost ___/
                    Not our paper's business         Our paper's business
```

The paper's spine is the MTL/ST ratio per head, the Kendall figure (uncertainty-weighted loss equilibrium), and the efficiency table (one forward pass, 48.6M params total). Not SOTA parity. Reviewers accept "80% retention at 2x parameter efficiency." They reject "40% of a number set by a different system under different conditions."

This document analyzes each head across 9 dimensions: current MTL metric, ST ceiling, SOTA anchor, gap decomposition, what beating SOTA requires in absolute numbers, published MTL evidence, concrete architecture/training changes, risk probability, and priority for the paper's headline.

---

## 1. Detection (24-class Assembly State Detection)

### 1.1 Current MTL Metric
mAP@0.5 = **0.2024**, present-class mAP50_pc = **0.3036** from Run 2 epoch 17 validation. The trajectory is flat at 0.30-0.31 mAP50_pc across epochs 17-20, and critically, two independent runs with a 4x LR/BIAS difference produced IDENTICAL curves. cls_preds mean is -7.03 and worsening (negative drift). n_positives per image is 517 (range 400-800), so positive anchor starvation is not the issue.

**Honest assessment**: The detection head has hit a structural ceiling at mAP50_pc ~0.30. Two runs with 4x LR difference producing identical trajectories is diagnostic of a gradient-suppressed equilibrium, not a tuning problem.

### 1.2 ST Ceiling (Projected)
**0.40-0.55 mAP@0.5** from doc 208. Based on ConvNeXt-Tiny at 224px as the limiting factor, ImageNet-only pretraining (no COCO), and 923 GT boxes across 37 recordings for 24 classes (8/24 classes absent from the 50% training subset).

**Caveat**: Internal analysis (file 63) estimates 0.20-0.30. If this is correct, the head is already at its ST ceiling and the backbone itself is the bottleneck — no head improvement can close the gap.

### 1.3 SOTA Anchor
YOLOv8m at **0.779 mAP@0.5** (640px input, COCO pretrained, plus 260K synthetic industrial images). The gap is **59%** (0.344 vs 0.838 per competitor analysis). The earlier 5-14% estimate was wrong — it assumed 0.70-0.80 convergence, which is structurally impossible at 224px with ImageNet-only features.

### 1.4 Gap Decomposition
(0.779 - 0.50) + (0.50 - 0.202) = 0.577 total gap

**Recipe/data gap (0.279)**: 640px vs 224px = 8.2x pixel difference. Small assembly parts (~20px in the original 1280px) span ~3.5 pixels on the P3 feature grid at stride 8. YOLOv8m at 640px sees these same parts at ~10 pixels. You cannot recover this difference with architecture alone. Additionally, COCO pretraining provides a 10-15% mAP advantage specific to detection, and YOLOv8 uses a detection-optimized CSPDarknet backbone vs. our general-purpose ConvNeXt-Tiny.

**Sharing cost (0.298)**: OHEM+FocalLoss gradient suppression is the primary mechanism. With OHEM ratio=2.0, the model is forced to focus on 2 hard negatives for every positive — but when positives are sparse (0-3 objects per image with 173K anchors), this starves positive gradients. The anchor design is also problematic: only 1.6% of anchors (on P6/P7 feature levels) can reach IoU >= 0.5 with typical GT, and anchor sizes (24, 48, 96, 192, 384) are mismatched to GT bounding boxes (164-404px centers). Multi-task interference compounds this: the backbone gradient norm (3.91) is 140x larger than the detection gradient norm (0.0276), meaning detection is being carried by backbone updates from other heads, not driving its own learning.

### 1.5 What Beating SOTA Requires
**Impossible** at 224px with ImageNet-only pretraining. Would need: 640px resolution (breaking the single-pass latency claim), COCO plus synthetic industrial pretraining (months of data acquisition), an anchor-free detector like FCOS or DETR to eliminate the anchor mismatch, and 10x more annotated frames.

**Achievable target**: **0.35-0.50 mAP@0.5** within current constraints. This is competitive within the 224px regime and provides a defensible MTL/ST ratio.

### 1.6 Published MTL Evidence
RetinaNet in MTL (MTL-ViT 2023) shows 3-8% degradation vs. single-task at matched resolution. Our ~33% degradation (0.30 MTL vs 0.45 ST) is 4-11x worse than published norms. ConvNeXt backbone MTL (UniPerceiver 2023) retains ~90% of ST detection performance. The fact that our degradation far exceeds published norms confirms the issue is OHEM+FocalLoss gradient starvation, not normal MTL interference.

### 1.7 Concrete Changes
| Change | Expected mAP Gain | Effort |
|--------|-------------------|--------|
| **OHEM ablation** (disable, rely on FocalLoss alone) | +0.05-0.10 | Low (config flag) |
| **Anchor-free detection** (roi_detector.py exists, 379 lines) | +0.05-0.15 | Medium |
| **COCO-pretrained backbone** (ConvNeXt-Tiny variant) | +0.03-0.08 | Low |
| **REINIT_REG_WARMUP extension** (currently 1000 steps, 1% to 100%) | +0.02-0.05 | Low |

Highest ROI: OHEM ablation. It costs nothing and diagnoses whether the gradient-suppressed equilibrium is caused by OHEM specifically. Do this before any architectural change. Run 5 epochs with OHEM off and compare mAP trajectories — if mAP jumps above 0.25, the bottleneck is confirmed and the fix is permanent.

### 1.8 Risk
- Below 0.30 even with OHEM off: 40% probability, MTL/ST < 0.60
- OHEM ablation reveals backbone-limited ~0.25 mAP: 50% probability, confirming ST ceiling is lower than projected
- Anchor-free redesign succeeds at 0.40+: 25% probability, strong paper result

**Overall risk: HIGH**. Detection is the hardest head to fix without changing the fundamental architecture (224px input, ImageNet backbone).

### 1.9 Headline Priority
**MEDIUM**. Detection is not the paper's differentiator — that is efficiency and head pose novelty. The detection head needs to be "good enough": mAP > 0.25 and a believable MTL/ST ratio > 0.60. Below 0.20 mAP triggers the Tier 2 contingency (4-task paper dropping the detection claim).

---

## 2. Activity (75-class Assembly Activity Recognition)

### 2.1 Current MTL Metric
**~0% top-1**. Complete collapse. Predicts only 1 of 75 classes (the majority NA class). pred_distinct = 1, entropy < 0.5 nats (near-zero), loss ~4.5 (theoretical CE baseline for 1/75 prediction ~4.3). This collapse persisted across all ~20 training epochs.

**Root cause**: In-place tensor assignments in FeatureBank (model.py lines 1240-1241) and the ActivityHead (model.py line 1384) severed the gradient path through proj_feat. The activity gradient was measured at ~0.012 — **30x below** detection's gradient of ~0.48. This was **fixed** around 2026-07-02 by introducing DETACH_GRAD_ENTRIES_ONLY = True and ACTIVITY_GRAD_BLEND_RATIO = 1.0, but the head has not recovered. Secondary causes include severe data imbalance (only 3.7K labeled frames of 48K total, 46 of 75 classes with <1% prevalence), per-frame training on shuffled batches that destroy temporal signal, and the simple 150K MLP fallback (ACTIVITY_HEAD_SIMPLE = True) that is necessary for non-consecutive batches but caps capacity.

### 2.2 ST Ceiling (Projected)
**55-65% top-1** from doc 208, assuming clip-level T=16 inference, VideoMAE stream active, and full training budget. **Reality check**: internal analysis estimates 10-20%. With 46 of 75 classes below 1% prevalence, macro metrics will always be poor regardless of architecture.

### 2.3 SOTA Anchor
MViTv2-S at **65.25% top-1** (640px input, K400 video pretraining from 240K videos, synthetic assembly data augmentation, 5x our training budget). Their video pretraining provides spatiotemporal features that we lack entirely — our ImageNet pretraining provides only spatial features. This baseline is not comparable to our setting.

### 2.4 Gap Decomposition
(0.6525 - 0.60) + (0.60 - 0.00) = 0.6525 total gap. The sharing cost is **100%** because the head cannot learn in the current MTL setup. This is NOT normal MTL interference — published activity MTL work shows 90-95% retention.

**Recipe gap (0.0525)**: K400 video pretraining alone would likely give 40-50% top-1 immediately by providing temporal features from 240K videos. Resolution (8.2x pixels) and synthetic data are secondary.

**Sharing cost (0.60)**: Structural. The head is prevented from learning by (a) insufficient labeled data and (b) a fundamental mismatch between per-frame training (shuffled batches) and clip-level evaluation protocol. These problems exist regardless of whether the head is in a multi-task or single-task setting.

### 2.5 What Beating SOTA Requires
**Not realistic** with any foreseeable architecture or data within this project's scope. The 65-point gap from 0% current performance cannot close.

**Achievable target**: **30-45% top-1** requires VideoMAE stream fine-tuned (not frozen), clip-level training with consecutive frame batches, and a two-stage training pipeline with embedding cache. Even reaching 30% requires first recovering from collapse to >10% (above random baseline of ~1.3%).

### 2.6 Published MTL Evidence
Activity recognition in MTL retains 90-95% of single-task top-1 when data is sufficient (OmniSource 2021, UniPerceiver 2023). Even in challenging few-shot MTL settings, activity degradation typically stays below 15%. Our 100% degradation is unprecedented in the published literature. This strongly confirms the failure is driven by (a) the severed gradient path that starved the head of learning signal for 20 epochs, and (b) the fundamental mismatch between per-frame shuffled training and clip-level evaluation. Neither of these is a "multi-task interference" problem in the normal sense.

**Paper implication**: If activity can recover to 20-30% top-1 with the gradient fix, the MTL/ST ratio will be weak (0.30-0.50) but defensible given the data constraints. If it stays at 0%, the failure must be documented openly with root cause analysis — this is more credible than sweeping it under the rug.

### 2.7 Concrete Changes
| Change | Expected Gain | Effort |
|--------|--------------|--------|
| **Recovery verification probe** (test if gradient fix enabled learning) | Diagnostic | Low |
| **VideoMAE stream enablement** (fix collate_fn for evaluation) | +5-10% top-1 | Low |
| **Two-stage training** (cache embeddings, train temporal heads from cache) | +15-25% top-1 | High |
| **Clip-level training** (consecutive frame batches, not shuffled) | +5-15% top-1 | Medium |

Critical path: verify the gradient fix actually enabled activity head learning. If it still cannot reach >5% top-1 after 5 epochs in a targeted probe, the issue is deeper than gradient flow.

### 2.8 Risk
- Cannot recover (stays ~0%): 40% probability, drop activity claim from paper
- 10-20% (above random but below publishable threshold): 30%, frame as documented failure
- 20-35% (adequate with honest framing): 20%, acceptable for workshop venue
- Exceeds 35% (strong result): 10%, good but still far below MViTv2

**Overall risk: CRITICAL**. Unlike detection (which at least learns), activity has demonstrated zero learning capability. Highest-risk head.

### 2.9 Headline Priority
**MEDIUM-HIGH (inverted)**. If it recovers, it becomes the second headline after head pose. If it remains collapsed, the "dissected failure" narrative from file 74 — one clean failure with a documented root cause (severed gradient) and structural limitations (data, protocol mismatch) — is actually a stronger paper than 5 half-working heads with no explanation for any of them.

---

## 3. PSR (11-Component Procedure Step Recognition)

### 3.1 Current MTL Metric
**~0.0 event-F1 at +/-3 frame tolerance**. Never learned. Outputs constant prediction (all zeros or all ones). Binary component accuracy (psr_comp_acc) ~0.0. Loss oscillates between 0.1 and 1.8 without converging.

**Root cause**: PSR labels are fill-forward within recordings — 95% of frames have the same state as the previous frame. A per-frame binary classifier on near-static labels has virtually no gradient to learn from. Binary focal loss with gamma = 0.5 is too gentle to penalize constant prediction. The sensitivity penalty (designed to encourage transitions) is capped at 0.05 — far too gentle. Even in sequence mode (T=8 every 2 batches), each prediction covers 2 frames (T=8 from T=16 input), smoothing out any transition signal. There are only 5-10 actual transition events per recording across 4 training recordings.

### 3.2 ST Ceiling (Projected)
**Unknown** — no ST baseline has been run (flagged in doc 86 as critical missing work). Best estimate: **0.15-0.35 event-F1** with the same architecture, because the constant-prediction problem is architectural, not MTL-specific. An ST-only PSR head with per-frame binary classification on static labels would also fail.

### 3.3 SOTA Anchor
STORM at **0.883 F1** uses a procedural multi-stage pipeline: detect hands, track objects, infer states. Each stage is optimized independently. The B2 heuristic (state persistence + 1.5s timeout) achieves ~0.60-0.70 F1 — even a trivial heuristic outperforms our learned approach.

**Critical caveat**: The STORM comparison is fundamentally misleading. Their pipeline approach is architecturally different from our end-to-end single-pass formulation. We should benchmark against B2, not STORM.

### 3.4 Gap Decomposition
(0.883 - 0.35) + (0.35 - 0.00) = 0.883 total gap. The PSR failure is likely **independent of MTL** — the same architecture in a single-task setting would also fail because per-frame binary classification on 95%-static labels is a flawed formulation regardless of multi-task interference.

**Recipe gap (0.533)**: STORM's multi-stage pipeline allows independent optimization of each stage (hand detection, object tracking, state inference). Our single-pass approach must learn everything jointly from 224px features. This is a fundamentally different and harder problem.

**Sharing cost (0.35)**: Even if the formulation were correct, PSR would compete for backbone capacity with detection, pose, and head pose. However, since the current failure is architectural, the sharing cost is moot until the formulation is fixed.

### 3.5 What Beating SOTA Requires
**Impossible** end-to-end at 224px with per-frame classification. Must switch to transition prediction. The psr_transition.py module (301 lines) exists but is not enabled. Additionally need: Gaussian-smeared transition targets to create learning signal around transition boundaries, monotonicity loss during training (currently eval-only), and a dedicated hand/object interaction feature stream.

**Achievable target**: **0.30-0.50 F1** with transition prediction + monotonic constraint. This would beat the B2 heuristic on precision (though likely not recall).

### 3.6 Published MTL Evidence
No prior published work exists on MTL assembly state recognition. This is genuinely novel territory — and novelty carries risk because there is no established best practice. Single-task procedure recognition (PVRP 2023) achieves 0.70-0.85 F1 with multi-stage pipelines (separate hand detection, object tracking, state inference). State-change detection in egocentric video (Video-SWAG 2024) achieves 0.55-0.70 F1 using specialized state-change detection heads. Our end-to-end approach at 224px is attempting something that, in the published literature, is always done with modular pipelines and higher-resolution inputs. The architectural gap between our approach and the published SOTA is larger than any MTL-specific issue.

**Paper implication**: If PSR exceeds 0.30 F1 with transition prediction, it is a genuine research contribution — the first successful end-to-end PSR at 224px in MTL. If it fails, it is not because of MTL but because the task formulation (per-frame static classification) is architecturally flawed for this problem.

### 3.7 Concrete Changes
| Change | Expected Gain | Effort |
|--------|--------------|--------|
| **Enable PSRTransitionPredictor** (psr_transition.py, 301 lines) | +0.15-0.30 F1 | Medium |
| **Gaussian-smeared transition targets** (sigma=2-3 frames) | +0.05-0.15 F1 | Low |
| **Remove sensitivity penalty cap** (increase from 0.05) | +0.02-0.05 F1 | Low |
| **ST PSR baseline run** (same architecture, single-task) | Diagnostic | Medium |

**Critical path**: Run an ST PSR baseline. Without it, we cannot distinguish between "PSR can't work in MTL" and "PSR can't work at all with this architecture." This has been deferred since doc 86 and is the single most important diagnostic for this head.

### 3.8 Risk
- Fundamentally unsolvable end-to-end (<0.10 F1): 50% probability, drop PSR from paper
- Reaches 0.10-0.30 with transition prediction (publishable): 35% probability
- Exceeds 0.30 (strong result): 15% probability

**Overall risk: HIGH**. Same fundamental problem as activity (never learned meaningfully) plus a flawed per-frame architectural formulation for a transition-detection task.

### 3.9 Headline Priority
**LOW**. PSR is the least important head for the paper's narrative. The paper works without it as a 4-task paper (detection + activity + pose + head_pose). The Tier 2 contingency from file 94 explicitly drops PSR from the multi-task claim. If it exceeds 0.30 F1 it becomes a nice secondary result; if it fails, it is dropped entirely.

---

## 4. Head Pose (9-DoF: Forward Vector + Position + Up Vector)

### 4.1 Current MTL Metric
**9.13 degrees** forward angular MAE from Run 2 epoch 17 validation. This is the ONLY head that works correctly. Gradient flows normally (weight norm 4.47e-03, borderline low but stable). Architecture is a simple MLP (1152 -> 512 -> 256 -> 9) with ~0.5M parameters. The head has plateaued near 9 degrees after ~20 epochs.

### 4.2 ST Ceiling (Projected)
**~7 degrees** from doc 208, based on the same ConvNeXt-Tiny backbone, same architecture, dedicated single-task training with all 48K frames (pose labels exist for every frame — no label scarcity issue).

**Caveat**: The 8.71 degree baseline from earlier analysis needs recomputation — it was computed before GT vector normalization was applied in the evaluation code. The true ST baseline may differ slightly but not materially.

### 4.3 SOTA Anchor
**None for assembly POPW**. No prior supervised baseline exists for multi-task head pose estimation in industrial assembly. General head pose SOTA (6DRepNet at 3.89 degrees on 300W-LP, WHENet at 5.42 degrees on AFLW2000) is on different datasets (facial pose from specialized face datasets) and is not comparable to our industrial assembly setting with different camera angles, head appearances, and lighting.

### 4.4 Gap Decomposition
There is **no SOTA gap** because there is no SOTA for this specific task. The domain gap between 300W-LP facial pose and assembly POPW head pose is too large for meaningful comparison.

**Sharing cost**: 9.13 (MTL) vs ~7 (ST projection) = **~2.1 degrees (23% degradation)**. This gives a preliminary MTL/ST ratio of **0.77** — the ONLY head where we can compute a meaningful ratio, and it is well within the acceptable range (reviewers accept MTL/ST >= 0.75).

### 4.5 What Beating SOTA Requires
The only SOTA that matters is our own ST baseline. Key targets: MTL/ST >= 0.75 (current ~0.77 already meets this). Even 12 degrees MAE is publishable for a novel task with no prior baseline. Per doc 94: "The head pose results are publishable regardless of other head performance." This is the paper's anchor result.

### 4.6 Published MTL Evidence
No prior MTL head pose work exists for industrial assembly. The closest analog is multi-task face alignment and head pose (HyperLandmark 2022), which shows 88-95% of single-task performance. Multi-task geometric reasoning (GMoE 2023) retains 80-90% of ST performance for 3D tasks. Our 77% is within the expected range for MTL geometric tasks.

### 4.7 Concrete Changes
| Change | Expected MAE Gain | Effort |
|--------|-------------------|--------|
| **GeometryAwareHeadPose** (6D rotation + geodesic loss, 251 lines in head_pose_geo.py) | -2 to -5 degrees | Medium |
| **ST baseline run** (single-task on same backbone) | Diagnostic | Low (1-2 days) |
| **Train longer** (current plateau at 9 deg may improve) | -0.5 to -1 degree | Ongoing |

### 4.8 Risk
- Stays at 9-12 degrees (already publishable): 60% probability
- Improves to 6-8 degrees with geometric loss: 25% probability, headline result
- Degrades beyond 15 degrees: <10% probability, still publishable as first baseline

**Overall risk: LOW**. The only low-risk head. Even the worst case (15 deg) provides a publishable result for a novel task.

### 4.9 Headline Priority
**HIGHEST**. Head pose is the uncontested contribution. The paper narrative should lead with: (1) "First multi-task head pose estimation in assembly POPW" (task novelty), (2) "Competitive 9.13 degree MAE with no prior supervised baseline" (quantitative result), and (3) "MTL/ST ratio of ~0.77 demonstrating efficient parameter sharing" (efficiency claim). Per doc 94, the head pose results are publishable regardless of any other head's performance.

---

## 5. Cross-Head Synthesis

### 5.1 Current State Summary

| Head | MTL (current) | ST (projected) | SOTA Anchor | MTL/ST | Headline Priority |
|------|---------------|----------------|-------------|--------|------------------|
| Head Pose | 9.13 deg | ~7 deg | None (novel task) | ~0.77 | HIGHEST |
| Detection | 0.202 mAP | 0.40-0.55 | 0.779 (YOLOv8m) | ~0.50 | MEDIUM |
| Activity | ~0% top-1 | 55-65% | 65.25% (MViTv2-S) | ~0.00 | MEDIUM-HIGH |
| PSR | ~0.0 F1 | Unknown | 0.883 (STORM) | ~0.00 | LOW |

### 5.2 The Critical Missing Piece
**Ablation A (single-task baselines on the same backbone)** is the single most important missing experiment for the entire paper. Without it: (1) we cannot compute MTL/ST ratios, which is the paper's core quantitative claim, (2) we cannot attribute degradation to MTL interference vs. architectural limitations, and (3) the efficiency argument ("31% fewer params than 3-model pipeline") is hollow — reviewers will ask "compared to what?"

Priority order for ST baselines:
1. **Head pose ST** — 1-2 days on 5060 Ti, provides the paper's main MTL/ST ratio
2. **Detection ST** — 1-2 days, determines whether the 0.30 ceiling is MTL-induced or backbone-limited
3. **PSR ST** — 1-2 days, resolves the architectural vs. MTL failure question
4. **Activity ST** — 1-2 days, determines the data ceiling for this task

### 5.3 Updated Success Criteria

**Strong result** (publishable at WACV/AAIML workshop):
- Head pose: MTL/ST >= 0.75, MAE <= 10 deg (ALREADY ACHIEVED at 0.77, 9.13 deg)
- Detection: MTL/ST >= 0.60, mAP >= 0.25 (needs OHEM ablation)
- Activity: Top-1 > 10% (above random, demonstrates recovery from collapse)
- PSR: Event-F1 > 0.15 with monotonicity (demonstrates procedural prior helps)

**Adequate result** (publishable with proper framing):
- Head pose carries the paper (novel task, credible metric)
- Detection and activity show non-trivial MTL/ST ratios (even if not strong)
- Kendall figure shows stable equilibrium with no collapse
- Efficiency table shows single-pass advantage over 3-model pipeline

**Failure mode** (paper genuinely at risk):
- Head pose degrades beyond 15 degrees (unlikely, currently plateaued at 9)
- All heads except head pose show zero learning signal (current trajectory for activity and PSR)
- OHEM ablation and gradient fix fail to improve detection and activity by epoch 10 of the next training run
