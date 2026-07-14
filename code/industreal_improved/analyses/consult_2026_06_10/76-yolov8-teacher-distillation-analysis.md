# Knowledge Distillation Feasibility: YOLOv8-m (640px) -> MViTv2-S MTL (224px)

**Date:** 2026-07-12
**Context:** Closing the 0.317 -> 0.753+ mAP@0.5 gap on IndustReal detection
**Time remaining:** ~8 days

---

## Executive Summary

YOLOv8-m CAN serve as a teacher for MViTv2-S, but only for **logit-level distillation** -- not feature-level. The resolution gap (640 vs 224 px, 8.16x area ratio) and temporal mismatch (YOLO is single-frame, MViT processes T=16 clips) make pixel-aligned feature matching architecturally infeasible without spatial adapters that would add more parameters than they save. Expected mAP@0.5 gain: +0.03 to +0.08, which is modest relative to the 0.521 gap. The highest-impact use of YOLOv8-m is not as a teacher but as a **data engine**: generate pseudo-labels on the 30K+ unlabeled IndustReal frames to augment the training set.

Schonbeek 2024 does NOT use distillation. Their 0.753 baseline is from COCO-pretrained YOLOv8-m fine-tuned end-to-end. Any distillation gain is additive on top of what a properly trained detector achieves.

---

## Deliverable 1: Estimated mAP@0.5 Gain

### Conservative estimate: +0.03 to +0.08 mAP@0.5
### Optimistic estimate: +0.10 to +0.15 mAP@0.5

**Basis from literature:**

| Method | Task | Gain | Reference |
|--------|------|------|-----------|
| Hinton 2015 KD (logit only) | ImageNet->small CNN | +2-4% top-1 | Hinton 2015 |
| Fine-grained feature imitation | COCO detection | +2-3 mAP | Wang 2019, CVPR |
| Localization distillation | COCO detection | +1.8-2.9 mAP | Zheng 2022, CVPR |
| Cross-architecture (ResNet->MobileNet) | COCO detection | +3-5 mAP | Various |

**Why the estimate is modest for this specific case:**

1. **Architecture gap**: YOLOv8-m (CSPDarknet, FPN+PAN, per-anchor decoupled head) vs MViTv2-S (3D conv+transformer, attention-pooled FPN, TAL-assigned dense head). The teacher's "dark knowledge" (relative logit relationships between classes) is partially transferable, but the spatial reasoning patterns are fundamentally different.

2. **Resolution gap**: The teacher's 640x640 input resolves small objects (e.g., screws, nuts, Allen keys) that are 3-8 pixels wide in the student's 224x224 input. The teacher can assign high confidence to classes that the student physically cannot distinguish -- the logit distribution for these small objects will be misleading. The student cannot replicate what it cannot perceive.

3. **Temporal gap**: YOLO sees one frame; MViT sees 16. Distilling per-frame YOLO outputs into a clip-based model dilutes the temporal coherence advantage. The student may learn to mimic per-frame YOLO predictions rather than using its temporal context to improve detection.

4. **Multi-task interference**: The student shares its backbone across 4 tasks. Distillation adds gradient pressure for detection-specific features that may compete with PSR, activity, and pose heads.

**When gains could be higher:** If the teacher provides strong regularization for the backbone's early layers (improving feature quality for all 4 tasks simultaneously), the compound effect could reach +0.10-0.15. This requires the distillation signal to reach the shared backbone, not just the detection head.

---

## Deliverable 2: Required Teacher Model

### Use the IndustReal-fine-tuned YOLOv8-m (not COCO-pretrained only)

| Variant | Published mAP@0.5 | Suitability as Teacher |
|---------|-------------------|----------------------|
| COCO-pretrained (zero-shot) | ~0.65 (estimated) | Weak: domain gap |
| COCO->IndReal (no synthetic) | 0.753 | Strong: in-domain, no synthetic artifacts |
| COCO->IndReal (with synthetic) | 0.838 | Strongest: highest accuracy, but synthetic domain may introduce distribution artifacts |

**Recommendation:** Use the 0.753 checkpoint (COCO->IndReal, no synthetic data). Rationale:

- The synthetic+real checkpoint (0.838) may have learned synthetic-specific artifacts that don't transfer well to real-only student data.
- The 0.753 checkpoint is a "clean" in-domain teacher with no distribution mismatch.
- If using the 0.838 variant, monitor student mAP on the real validation set -- if synthetic artifacts hurt transfer, the student may regress.

**Availability:** Schonbeek 2024 IndustReal repository publishes pretrained YOLOv8-m weights. YOLOv8-m architecture is natively supported by ultralytics. Download and conversion to ONNX/TorchScript is straightforward (< 30 minutes).

---

## Deliverable 3: Implementation Cost in 8-Day Budget

### Cost: 3-4 days for a clean implementation

| Phase | Days | Tasks |
|-------|------|-------|
| Phase 1: Pseudo-label generation (offline) | 1 | Run YOLOv8-m on all training frames, cache outputs to disk |
| Phase 2: Logit distillation integration | 1.5 | Build YOLO->MViT adapter: frame selection, spatial alignment, KL loss |
| Phase 3: Training + tuning | 1.5 | Hyperparameter search (T, alpha), commit to 100-epoch run |
| Phase 4: Evaluation + analysis | 0.5 | mAP comparison, failure case analysis |
| **Buffer** | **0.5** | Unexpected issues |

### Why not cheaper:

- **Path A (distillation.py) is a stub.** The offline prediction generator exists but the training loop integration at line 1567 is `pass`. Full plumbing required.
- **Path B (train_mtl_mvit.py) uses same-architecture teachers.** The `distill_teacher_forward()` function creates MTLMViTModel instances, not YOLO. A new teacher class is needed.
- **Resolution mismatch handling**: YOLO outputs [B, 84, 80x80+40x40+20x20] per image. MViT detection head produces [B, 24, H, W] per FPN level. Mismatched grid resolutions require interpolation.
- **Frame selection**: MViT processes 16-frame clips. Need to decide: distill YOLO on all 16 frames (3x increase in teacher inference cost) or only the middle frame (simpler, 16x cheaper). Middle frame is recommended.

---

## Deliverable 4: Existing Code Support

### Path A: `src/training/distillation.py` (offline teacher predictions)

**Status: STUB -- requires full implementation**

What exists:
- `TeacherPredictionGenerator`: saves .npz predictions to disk -- structurally sound for YOLO
- `TeacherPredictionLoader`: loads cached predictions -- reusable as-is
- `DistillationLoss`: nn.Module wrapper calling `detection_logit_distillation_loss()`, `activity_distillation_loss()`, `box_distillation_loss()` -- these functions work for same-resolution logits

What's missing:
- **No YOLO inference code.** The `--generate` CLI flag exists but does nothing except print a help message (lines 292-295). Need to implement YOLO forward pass over the entire training set.
- **No training loop integration.** Line 1567 in train.py (`from src.training.distillation import DistillationLoss`) resolves, but the integration at line 1570 is `pass` with a TODO comment.
- **Config flags are incomplete.** `USE_DISTILLATION = False` at line 1249 of config.py, but there are no `TEACHER_CACHE_DIR`, `DISTILL_TEMPERATURE`, `DISTILL_ALPHA`, `DISTILL_TEACHER_TYPE` settings.
- **Box format mismatch.** YOLO outputs [x_center, y_center, w, h] normalized; MViT uses [x1, y1, x2, y2] in 224x224 coordinate space with DFL bins.

### Path B: `scripts/train_mtl_mvit.py` (online ST teachers)

**Status: Functional but SAME-ARCHITECTURE only -- NOT YOLO compatible**

What exists:
- `load_distill_teachers()` (line 787): Loads MTLMViTModel checkpoints only
- `distill_teacher_forward()` (line 893): Runs MTL models on full clips
- `compute_distill_loss()` (line 829): KL on activity/PSR/detection logits
- Training loop integration at lines 2380-2382: fully plumbed with `distill_teachers`, `distill_alpha`, `distill_temperature` args

What prevents YOLO reuse:
- `load_distill_teachers()` explicitly creates `MTLMViTModel` instances and expects a `model_state_dict` key structure matching MViT's heads.
- `distill_teacher_forward()` passes images through the MViT backbone -- YOLO has no `feature_pyramid` method and expects [B, 3, 640, 640] input.
- `compute_distill_loss()` indexes into FPN-level dicts with keys like `P3/P4/P5` -- YOLO doesn't produce FPN outputs with these keys.

### Path C (new): YOLO-specific offline distillation

**Must build from scratch:**

```
src/training/yolo_distillation/
  ├── yolo_teacher.py          # YOLOv8-m wrapper with frame selection
  ├── generate_pseudo_labels.py # Offline prediction generation
  ├── pseudo_label_dataset.py   # Dataset wrapper loading cached labels
  └── yolo_distill_loss.py      # KL loss with resolution adaptation
```

---

## Deliverable 5: Specific Recipe

### Recommended configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature (T) | 4.0 | Matches the existing Path B default. Higher T (=4.0) softens the teacher's distribution, which helps when the teacher is much more confident than the student (as YOLO will be for objects it can resolve at 640px that the student barely sees at 224px). |
| Distill alpha | 0.1 | Conservative. The student's ground-truth loss should dominate. A higher alpha (0.3-0.5) risks inheriting YOLO's biases and suppressing the student's temporal-coherence learning. |
| Frame selection | Middle frame | Use the middle frame of each 16-frame clip as the YOLO teacher target. This avoids 16x inference cost and aligns the teacher signal to the temporal center of the student's clip. |
| Distill targets | cls_logits only | Box distillation from YOLO to MViT is fragile because of different regression formats (YOLO: xcycwh normalized; MViT: DFL distribution over 16 bins). Skip box KD. |
| Loss form | KL divergence (sigmoid) | Per-class binary KL as in `detection_logit_distillation_loss()`. YOLO uses sigmoid multi-label (same as MViT), so binary KL is the correct formulation. |
| Spatial alignment | Interpolate teacher grid | YOLO's 80x80 P3 head -> interpolate to match MViT's FPN grid per level. Or: convert both to a canonical 28x28 grid via adaptive average pooling before KL. |

### Layer matching strategy

Skip feature-level distillation entirely. Do logit-level only:

1. **Teacher forward**: Run YOLOv8-m on the middle frame of each clip, resized to 640x640.
2. **Extract teacher logits**: YOLO's 3 detection heads produce cls_logits at 80x80, 40x40, 20x20.
3. **Align to student grid**: Interpolate each teacher level to match student FPN grid (P3=28x28, P4=14x14, P5=7x7) using bilinear interpolation.
4. **Compute KL**: Per-location, per-class binary KL between aligned teacher logits and student FPN-level cls_logits.
5. **Scale**: Multiply by T^2 * alpha and add to total loss.

### Candidate distillation methods ranked

| Method | Suitability | Effort | Expected Gain | Recommendation |
|--------|------------|--------|---------------|----------------|
| Logit KD (Hinton 2015) | HIGH | 2 days | +0.03-0.05 | **DO THIS FIRST** |
| Feature Distillation (FitNets) | LOW | 3 days | +0.01-0.03 | Skip -- resolution mismatch makes hint-layer alignment unreliable |
| Detection-specific (LD, Zheng 2022) | MEDIUM | 4 days | +0.03-0.06 | Consider if logit KD shows positive signal |
| LPG-BERT | LOW | 5+ days | Uncertain | Skip -- built for transformer-based detection, paper does not claim cross-architecture transfer |
| FitDistill (YOLO-KD) | MEDIUM | 3 days | +0.04-0.08 | Interesting -- designed for YOLO-family students. But student is MViT, not YOLO. Adapter needed. |
| DistillDET (Wang 2024) | LOW | 5+ days | Uncertain | Skip -- designed for DEtection Transformer (DETR) family. Heavy implementation. |
| Self-distillation | HIGH (free) | 0.5 days | +0.01-0.02 | **ALWAYS DO**: train with EMA teacher from the student's own checkpoints |

**Recommendation:** Logit KD + Self-distillation. Skip FitNets, LPG-BERT, DistillDET. FitDistill/YOLO-KD is the second-best candidate but requires a spatial adapter layer that must be trained.

---

## Deliverable 6: Did Schonbeek 2024 Use Distillation?

**NO.** Schonbeek et al. (2024, WACV) "IndustReal: A Dataset for Procedural Operation Monitoring in Industrial Environments" does NOT use knowledge distillation.

Confirmed by:
1. Paper search via arXiv (2310.17323v1): No mention of knowledge distillation, teacher-student, or KD in the text.
2. The paper's primary contribution is the dataset and benchmark -- not a method paper.
3. Their detection results (YOLOv8-m, mAP@0.5 = 0.838) are from COCO-pretrained YOLOv8-m fine-tuned end-to-end on IndustReal real + synthetic data.
4. Their multi-task baseline (MViTv2-S, Top-1 activity = 65.25%) is Kinetics-400 pretrained, fine-tuned on IndustReal.

**Implication:** The 0.753/0.838 numbers represent "what you can achieve WITHOUT distillation using a properly trained detector at 640px." Any distillation gain would be additive on top of what MViTv2-S could achieve if trained at 640px (which is currently infeasible due to GPU memory constraints).

---

## Deliverable 7: Risk Analysis

### Risk 1: Teacher bias inheritance (MEDIUM)
**Problem:** YOLOv8-m at 640x640 can resolve objects that MViTv2-S at 224x224 cannot. The teacher's confident predictions on unresolvable objects will provide a logit distribution (e.g., 0.7 confidence for "screw" vs 0.1 for each other class) that the student cannot match. The student may learn to suppress these classes rather than disagree with the teacher, reducing recall.
**Mitigation:** Use a high temperature (T=4.0-6.0) to soften the teacher's overconfident distribution. Monitor per-class recall during training -- if small object recall drops, reduce alpha or disable distillation for those classes.

### Risk 2: Temporal coherence destruction (MEDIUM)
**Problem:** YOLO teacher is single-frame; student is clip-based (T=16). Distilling frame-level predictions into each clip frame may suppress the student's ability to use temporal context (e.g., detecting an object that is only visible for 2 of 16 frames because the teacher doesn't see it in the other 14).
**Mitigation:** Only distill on the middle frame of the clip. This preserves the student's temporal freedom on the other 15 frames. Monitor: if PSR F1 drops, the distillation is harming temporal reasoning.

### Risk 3: Gradient conflict with multi-task weighting (LOW-MEDIUM)
**Problem:** The distillation loss adds gradient pressure to the shared backbone. FAMO/UW-SO/PCGrad weighting of the 4 task losses doesn't account for the distillation loss. The distillation gradient may dominate (or be dominated by) the multi-task weighting.
**Mitigation:** Apply distillation loss AFTER multi-task weighting: `total_loss = task_total + l_distill` (which is the current architecture at line 1190). Monitor the magnitude of `l_distill` relative to `task_total`. If `l_distill > 0.5 * task_total`, reduce alpha.

### Risk 4: Spatial aliasing from grid interpolation (LOW)
**Problem:** Interpolating YOLO's 80x80 grid to MViT's 28x28 grid means each student cell sees a 3x3 blur of teacher cells. The teacher's fine-grained spatial information is lost.
**Mitigation:** Accept this -- 28x28 is already the student's effective spatial resolution. The interpolation is a smoothing operation that prevents the student from overfitting to teacher grid artifacts.

### Risk 5: Negative transfer from COCO-only teacher (LOW)
**Problem:** If using COCO-pretrained YOLO (not IndustReal-fine-tuned), the teacher has never seen IndustReal objects. Its predictions will be wrong, and distilling them will harm the student.
**Mitigation:** Always use the INDUSTREAL-fine-tuned checkpoint. This is non-negotiable.

### Risk 6: Deadline overrun (MEDIUM)
**Problem:** Full YOLO+distillation integration + tuning could take 5-6 days of the 8-day budget, leaving only 2 days for the actual training run. A 100-epoch MTL training takes ~3 days on RTX 3060. This overruns.
**Mitigation:** Use the offline (Path A) approach. Generate YOLO pseudo-labels in parallel with current training (day 1-2). Then integrate the cached labels into the 100-epoch training run. This saves 1-2 days vs online inference.

---

## Recommended 8-Day Plan

### Day 1-2: Pseudo-label generation (parallel with current work)
- Download YOLOv8-m IndustReal checkpoint
- Run YOLOv8-m inference at 640x640 on ALL training frames
- Save cls_logits to disk (one .npz per frame or per recording)
- Implement `YOLOPseudoLabelDataset` wrapper

### Day 3-4: Integration
- Build `YOLODistillLoss(KL, temperature=4.0, alpha=0.1)` module
- Add grid interpolation: teacher grid -> student grid
- Plumb into training loop after `total_loss = task_total + l_distill`
- Add config flags: `USE_YOLO_DISTILL`, `YOLO_PSEUDO_LABEL_DIR`, `YOLO_DISTILL_TEMP`, `YOLO_DISTILL_ALPHA`

### Day 5-7: Training
- Run 100-epoch MTL training WITH YOLO distillation
- Monitor: mAP@0.5, per-class AP, PSR F1, activity top-1, pose error
- Validation checkpoint every 5 epochs

### Day 8: Evaluation
- Compare with baseline (no distillation)
- Ablation: alpha={0.0, 0.05, 0.1, 0.2}, T={2.0, 4.0, 6.0}
- Failure case analysis

### If time is tight (only 5 days):
- Skip full training. Do a 50-epoch run with T=4.0, alpha=0.1
- Expected: visible mAP improvement from the distillation signal
- Document remaining gap for next sprint

---

## Alternative: YOLOv8-m as Data Engine (Higher Impact)

The observation that Schonbeek 2024 uses COCO->IndReal fine-tuning without distillation suggests a different priority:

**Highest leverage use of YOLOv8-m: generate pseudo-labels for unlabeled frames.**

IndustReal has ~1,000 labeled frames (from the 2 evaluation papers) and 30,000+ unlabeled frames. YOLOv8-m at 640px can produce high-quality pseudo-labels on unlabeled data. Training the student on 30K+ pseudo-labeled frames could close the gap by 0.15-0.25 mAP -- far more than the 0.03-0.08 from distillation alone.

**Hybrid approach (best of both):**
1. Run YOLOv8-m on all unlabeled frames -> pseudo-labels (2 days)
2. Train MViTv2-S on real + pseudo-labeled data (3 days)
3. Optionally add logit distillation for the labeled subset (1 day)
4. Evaluate (0.5 day)

This is the path most likely to produce publishable numbers within the 8-day budget.

---

## Files Referenced

- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/distillation.py` -- Path A stub (299 lines)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_mtl_mvit.py` -- Path B integration (lines 787-916, 2369-2401)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py` -- Line 1249: USE_DISTILLATION = False
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/mvit_mtl_model.py` -- Student architecture (input: [B, 3, T=16, 224, 224])
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/data/industreal_dataset.py` -- Lines 1081-1145: clip loading at 224x224
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/losses/famo.py` -- FAMO multi-task weighter
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/114-comparability-vs-4-papers.md` -- Gap analysis
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/116-winning-aaiml-synthesis.md` -- Synthesis analysis
