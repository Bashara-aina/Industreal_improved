# 125 — 50 Truly Deep Questions to Beat Every SOTA Result

**Generated:** 2026-07-04
**Scope:** 50 questions across 10 categories (5 each) designed to surpass every published SOTA result on the IndustReal dataset. Each question targets a specific architectural, algorithmic, or data gap and proposes a concrete experiment that would move the project from 'competitive' to 'beats SOTA on every track.'

**Reference papers (SOTA targets):**
- Paper 1 (YOLOv8m): detection 0.838 mAP@0.5, synthetic-data-augmented ASD detection
- Paper 2 (MViTv2): activity 65.25% Top-1 on 75 classes, Kinetics-400 pretrained, 16-frame temporal
- Paper 3 (B3): PSR 0.797/0.883 POS/F1, transformer-based, 22.4s tau
- Paper 4 (STORM-PSR): PSR 0.812/0.901 POS/F1, multi-modal, 15.5s tau

**Current bests (from 111-overview.md Section 4, 112-training-metrics-deep-dive.md Section 8):**
- Detection: mAP50=0.317, mAP50_pc=0.506 (epoch 11, still climbing per 118 §1)
- Activity: macro-F1=0.110, Top-5=0.398 (per-frame, zero temporal)
- Ego-pose: fwd MAE=8.14 deg, up MAE=7.06 deg (first baseline, no prior)
- PSR: POS=0.968, F1@3=0.144, edit=0.752, comp_acc=0.346

**SOTA gaps to close:**
- Detection: 0.317 vs 0.838 = 62% gap (or 0.506 vs 0.838 = 40% gap on present-class metric)
- Activity: 0.110 vs 0.25-0.35 (remapped MViTv2 estimate per 117 Q45) = 55-69% gap
- PSR F1: 0.144 vs 0.901 = 84% gap
- Ego-pose: no prior, must beat own baseline toward HL2 floor (~5-7 deg)

---

# Category 1: Architecture Changes (5 Questions)

Current architecture: ConvNeXt-Tiny backbone (28.6M), FPN neck (4.5M), RetinaNet detection head (5.3M), MLP activity head (0.7M), ego-pose head (2.8M with FiLM), MonotonicDecoder PSR (3.1M). Total 46.5M params on RTX 5060 Ti 16GB. These five questions explore backbone swaps, neck redesigns, attention modules, and head architecture changes that could close the detection gap from 0.317 toward 0.6+ without changing the multi-task formulation.

---

### Q1. ConvNeXt-V2 Backbone with FCMAE Pretraining

**Context:** Current ConvNeXt-Tiny backbone is randomly initialized (117 Q26) because early experiments showed -0.02 mAP regression with ImageNet-1K weights under 5e-4 backbone LR (115-execution-plan-to-sota.md:102). ConvNeXt-V2 (Woo et al., CVPR 2023) introduces FCMAE (fully convolutional masked autoencoder) pretraining that achieves 82.5% ImageNet Top-1 for Tiny variant with improved dense-task transfer. FCMAE's all-convolutional architecture produces spatially dense features at every stage — directly beneficial for ASD where fine-grained spatial details (washer-on-bolt vs washer-next-to-bolt) determine the 24-class state. The current random-init backbone is the single largest architectural bottleneck: every detection head, activity head, and PSR decoder depends on ConvNeXt features that have never seen any structured visual training.

**Question:** Would ConvNeXt-V2-Tiny with FCMAE pretrained weights, using a discriminative LR schedule (frozen backbone 5 epochs at 1e-5, unfrozen at 5e-5 per 117 Q26's proposal), improve det_mAP50 from 0.317 to 0.38-0.42 within 25 epochs — and by what margin does the FCMAE pretraining outperform standard ImageNet-1K ConvNeXt-V1 weights?

**Why this matters:** FCMAE is architecturally matched to the dense prediction tasks (detection, pose) that dominate 57% of our current gradient budget (112-training-metrics-deep-dive.md:883-888). Unlike ViT-based MAE, FCMAE preserves spatial resolution at every layer through full-conv design. If this closes detection gap by 0.06-0.10 mAP, it is the single most impactful architecture change available. Paper 1 uses YOLOv8m with COCO pretrain; no IndustReal published work uses ConvNeXt-V2 or FCMAE. Estimated impact: +0.06-0.10 mAP50, +0.03-0.05 mAP50_pc, activity +0.01-0.02.

**Constraints:** Must verify FCMAE weights are available in timm. 50-epoch full run on 5060 Ti (~7 days). Cannot freeze backbone past epoch 5 or the activity/pose heads will not converge. VRAM increase negligible (< 200 MB for same parameter count).

**Hypothesis:** FCMAE-pretrained ConvNeXt-V2-Tiny achieves det_mAP50=0.38-0.42 at epoch 25 (vs 0.317 at epoch 11 for random init). Activity macro-F1 reaches 0.12-0.14. The pretraining benefit is concentrated on low-GT channels (16, 19, 22) showing +0.08-0.15 AP each, because FCMAE provides structured feature representations that partially compensate for sparse detection supervision.

**Validation:** Download timm ConvNeXt-V2-Tiny FCMAE weights. Run full multi-task training for 50 epochs with identical config except: PRETRAINED=True, backbone discrim LR schedule. Compare epoch-25 vs epoch-25 of an identically-configured random-init run. Success criterion: det_mAP50 >= 0.38 at epoch 25.

---

### Q2. DyHead: Unified Attention Head Across All Four Tasks

**Context:** Each task head operates independently on shared FPN features (112-training-metrics-deep-dive.md:198-210). Detection uses RetinaNet cls+reg subnets, activity uses MLP, pose uses FiLM+MLP, PSR uses MonotonicDecoder. DyHead (Dai et al., CVPR 2021) replaces per-task heads with a unified attention mechanism that applies scale-aware, spatial-aware, and task-aware attention sequentially — enabling the model to dynamically allocate representational capacity across tasks. DyHead has demonstrated 1.5-2.5 AP improvement on COCO detection when replacing RetinaNet heads. On a 4-task system with imbalanced gradient budgets (det 27.2%, pose 27.2%, act 14.8%, psr 30.7%), the static head allocation is fundamentally mismatched to the dynamic task demands.

**Question:** Would replacing the four independent task heads with a single DyHead module (adding ~3M params, removing ~4M task-specific params for net -1M) improve combined metric from 0.363 to 0.40-0.45 by enabling dynamic feature allocation — specifically, would DyHead's task-aware attention mechanism allow the activity head (currently 14.8% gradient share) to borrow representational capacity from the larger detection and pose gradient streams?

**Why this matters:** The current heads are individually sized for their tasks (det 5.3M, act 0.7M, pose 2.8M, psr 3.1M) but the fixed allocation means the activity head is perpetually capacity-starved at 0.7M params for 69 classes. DyHead shares capacity dynamically, potentially giving activity more effective parameters at inference without increasing training gradient conflict. Paper 1-4 use standard per-task heads; no prior IndustReal work uses unified attention. Estimated impact: +0.02-0.04 combined, activity macro-F1 +0.02-0.04, detection flat to +0.01.

**Constraints:** Code change to model.py (~150 lines for DyHead integration). Adds SE-like attention overhead per FPN level. Must profile VRAM: estimated +1GB at B=4. Requires 50-epoch full run on 5060 Ti. Cannot be mid-run swapped; requires fresh training.

**Hypothesis:** DyHead achieves combined metric 0.40-0.45 (vs 0.363) with largest per-task gains on activity (+0.03 macro-F1) and small detection gain (+0.01 mAP50). The unified attention is most beneficial for the capacity-starved activity head, less so for already-capable detection and pose heads.

**Validation:** Implement DyHead as unified head. Remove per-task heads. Run 50 epochs. Compare all 4 task metrics against per-task-head baseline at equivalent epochs. Success criterion: combined metric >= 0.40 at epoch 25.

---

### Q3. Cross-Scale Transformer (CST) Neck: Replacing FPN with Hierarchical Attention

**Context:** Current FPN neck (4.5M params) received near-zero gradient for epochs 1-11 due to F1 gradient wipe bug (113-all-fixes-chronicle.md:460-505). Even post-fix, FPN's top-down pathway has limited bottom-up information flow. Cross-Scale Transformer (CST) replaces FPN with a transformer that attends across all feature pyramid levels simultaneously, computing cross-scale affinities that let fine-grained details from P3 directly influence coarse semantics at P5 and vice versa. The ASD taxonomy has objects at multiple scales simultaneously: a bolt (small, ~30px) and a washer (medium, ~80px) on the same assembly. FPN processes these through separate pathways; CST allows them to exchange information.

**Context from binary code clustering (116-winning-aaiml-synthesis.md:171-186):** Channels 9-12 form a 1-2 bit cluster where AP ranges from 0.368 to 0.886. These channels differ by exactly which components are present/absent — a cross-scale reasoning problem (is the washer present at P4 scale? is the bolt present at P3 scale?). CST is architecturally designed for exactly this cross-scale correspondence problem.

**Question:** Would replacing the 4.5M-param FPN neck with a Cross-Scale Transformer (2 transformer layers, 4 heads, 256 dim, ~3M params) close the 1-2 bit class confusion gap — specifically bringing channels 12 (0.368 AP) and 22 (0.063 AP) to within 50% of their high-AP neighbors — by enabling cross-scale feature correspondence that FPN's top-down-only information flow cannot provide?

**Why this matters:** The FPN was the single most severely damage-affected component (11 epochs of near-zero gradient). Even post-fix, its architectural limitation (top-down only) may be insufficient for the ASD binary-code reasoning task. CST directly addresses the binary-code confusion by allowing the model to explicitly correspond features across scales. Paper 1 uses CSP-PAN whose cross-stage connections are simpler than full cross-scale attention. Estimated impact: +0.03-0.06 mAP50_pc, concentrated on 1-2 bit confusion pairs.

**Constraints:** Code change to model.py. 2 transformer layers add ~800 MB VRAM. 50-epoch run on 5060 Ti. Must test at B=2 if VRAM exceeds 15 GB. Inference speed decrease: ~15-20% (transformer self-attention over pyramid positions).

**Hypothesis:** CST improves mAP50_pc from 0.506 to 0.55-0.58. Channels 12 (+0.10-0.20) and 22 (+0.06-0.15) show strongest improvement because cross-scale attention resolves their confusion with high-AP neighbors. PSR F1 improves +0.02-0.05 due to better ASD feature quality.

**Validation:** Implement CST neck. Run 50 epochs. Compare per-class AP (focus on channels 9-12 cluster, 22, 16, 19) against FPN baseline. Success criterion: mAP50_pc >= 0.55 at epoch 50.

---

### Q4. Sparse Conditional Detection Head: YOLOX-Style Decoupled Head with OTA Label Assignment

**Context:** Current RetinaNet detection head uses fixed anchor assignment with IoU-based matching and OHEM. YOLOX (Ge et al., 2021) introduced a decoupled head (cls + reg separate convs) with SimOTA label assignment (dynamic positive/negative allocation based on cost). The fixed anchor assignment in our RetinaNet head may be suboptimal for the 24-class ASD problem where ground-truth boxes are sparse (4,710 frames) and class ratios vary 15x (ch7=74 instances, ch21=5 instances). SimOTA dynamically assigns anchors to ground-truth based on classification + regression cost, allowing rare classes to claim more anchors than the fixed IoU-based assignment.

**Additionally:** The current OHEM+FocalLoss double suppression (117 Q5) may be entirely bypassed by SimOTA, since SimOTA replaces OHEM with dynamic cost-based assignment. Q5 hypothesizes that removing OHEM with gamma_neg=2.0 recovers rare classes. SimOTA could achieve the same effect without any loss-function changes.

**Question:** Would a YOLOX-style decoupled detection head (separate cls/reg convs, 3x3 depthwise + 1x1, ~2.5M params) with SimOTA label assignment (dynamic positive anchor count per GT, cost-based rather than IoU-based) improve mAP50 from 0.317 to 0.36-0.40 and mAP50_pc from 0.506 to 0.55-0.60 — and would the effect on rare classes (ch16, 19, 22 with AP near zero) be larger than the Q5 OHEM-removal experiment because SimOTA provides a more fundamental fix to the anchor assignment problem?

**Why this matters:** YOLOX's SimOTA has become standard in modern detectors precisely because fixed anchor assignment fails on long-tail data. Our 15x class imbalance is exactly the regime where SimOTA should provide the largest gains. If this succeeds, it simultaneously replaces OHEM (one less hyperparameter to tune) and provides a clean comparison to YOLOv8m's TaskAlignedAssigner. Estimated impact: +0.04-0.08 mAP50, +0.04-0.09 mAP50_pc.

**Constraints:** Code change to roi_detector.py (~200 lines for decoupled head + SimOTA). 50-epoch run. VRAM impact: < 200 MB added. Must verify SimOTA stable with batch_size=4 (designed for larger batches — may need to adjust cost threshold). Cannot run mid-training; requires fresh run.

**Hypothesis:** SimOTA head achieves mAP50=0.36-0.40 and mAP50_pc=0.55-0.60. Channels 16, 19, 22 recover to AP > 0.05 (vs 0.000, 0.000, 0.063) because dynamic anchor allocation provides them with sufficient positive supervision. Top channels (7, 9, 10) stay within 0.02 of baseline.

**Validation:** Implement decoupled head + SimOTA. Run 50 epochs. Compare per-class AP, mAP50, mAP50_pc. Success criterion: mAP50_pc >= 0.55 at epoch 25.

---

### Q5. Multi-Scale Feature Pyramid with FPN+PAN+NAS-FPN Ensemble

**Context:** Current single FPN neck (4.5M params, 112-training-metrics-deep-dive.md:198). YOLOv8m uses FPN+PAN (Path Aggregation Network) with CSP connections. NAS-FPN (Ghiasi et al., CVPR 2019) uses neural architecture search to find optimal cross-scale connections. The strong binary-code clustering in ASD (channels 9-12 differ by 1-2 bits, 116-winning-aaiml-synthesis.md:171-186) suggests that feature pyramids need bidirectional information flow — bottom-up to detect small components (bolts, washers) and top-down to recognize their assembly state. A single FPN does one direction; PAN adds the second; NAS-FPN finds the optimal routing between all scales.

**The key insight from binary-code structure:** Channel 22 (binary 11101111111, AP=0.063) differs from channel 10 (11110111110, AP=0.872) by exactly 2 bits — component 4 absent/present and component 6 absent/present. Detecting component 6 requires bottom-up detail (small feature at P3); detecting the assembly state requires top-down semantics (state context at P5). A unidirectional FPN can only do one at a time.

**Question:** Would a NAS-FPN-style learnable cross-scale routing (adding ~2M params to the current 4.5M FPN, total 6.5M neck) improve mAP50_pc from 0.506 to 0.56-0.61 by learning optimal bidirectional information flow paths for each ASD channel — specifically learning that channel 22 needs a direct P3-P5 shortcut to simultaneously detect component 6 (small) and recognize the overall assembly state (large context)?

**Why this matters:** NAS-FPN was the highest-performing neck on COCO at time of publication (+2-3 AP over FPN). For the ASD binary-code taxonomy where individual components have different optimal scales, learnable routing could provide substantial gains. No IndustReal paper uses NAS-FPN. Estimated impact: +0.05-0.10 mAP50_pc.

**Constraints:** Code change to model.py for NAS-FPN routing module (~100 lines). 50-epoch run on 5060 Ti. The routing weights themselves are 0.1M parameters; the additional feature transformations add ~2M. VRAM increase ~1 GB at B=4. Cannot be run mid-training.

**Hypothesis:** NAS-FPN neck achieves mAP50_pc=0.56-0.61 (vs 0.506 baseline) at epoch 50. The learned routing weights show high attention on cross-scale connections for channels 22, 16, 19 (transitional states with small-component differences). PSR F1 improves +0.03-0.06 due to higher-quality feature inputs.

**Validation:** Implement NAS-FPN-style neck with learnable routing parameters. Run 50 epochs. Compare per-class AP, mAP50_pc, PSR F1. Success criterion: mAP50_pc >= 0.56 at epoch 50.

---

# Category 2: Training Recipe Changes (5 Questions)

Current recipe: OneCycleLR (peak_factor=0.5), EMA (decay=0.995), AdamW (lr=2.5e-4 head, 2.5e-5 backbone), batch_size=8 (4 per GPU x 2 GPUs effective), mixup disabled, label_smoothing=0.1 cls / 0.0 act, 100 epochs, no SWA, bf16 disabled. These questions explore schedule modifications, optimizer changes, regularization, and training-time algorithms.

---

### Q6. Progressive Learning Schedule: Curriculum from Easy to Hard States

**Context:** The 24 ASD states follow an inherent curriculum from natural assembly order: earlier states (channels 0-5, "nothing present" through "base plate attached") are visually simpler with higher AP (0.51-0.89) while later states (channels 16-23, transitional/subtractive) have AP ranging 0.000-0.22 (111-overview.md:175-190, per-class AP table). The current OneCycleLR treats all epochs identically in terms of state difficulty. Progressive learning (Bengio et al., 2009; Weinshall et al., 2018) trains on easier examples first and gradually introduces harder ones — demonstrated to improve long-tail performance by 3-5% on ImageNet-LT.

**Context from assembly state liveness (112-training-metrics-deep-dive.md Appendix B):** Rare components h4 (19.1% prevalence), h7-h9 (low single digits) have gradient RMS < 0.005 through epoch 12. These components may never "escape" their low-gradient region under uniform sampling because they appear in too few batches to accumulate sufficient updates.

**Question:** Would a curriculum schedule that trains on easy states (channels 0-7, natural assembly order) for epochs 1-10, introduces medium states (channels 8-15) at epochs 11-20, and adds hard/transitional states (channels 16-23) at epochs 21-100 improve per-class AP for channels 16/19/22 from near-zero to 0.05-0.15, while maintaining top-channel AP — and would the mechanism be that early curriculum provides the backbone with strong feature representations before being asked to distinguish subtle 1-2 bit state differences?

**Why this matters:** The binary-code clustering analysis (116-winning-aaiml-synthesis.md:171-186) shows that channels 9-12 form a confusion cluster. A curriculum that first establishes clean prototypes for the easy states, then progressively adds harder distinctions, maps directly to the inherent difficulty ordering in the ASD taxonomy. This is a novel approach for assembly state detection; no IndustReal paper uses curriculum learning. Estimated impact: +0.02-0.05 mAP50_pc on easy states flat, hard states +0.05-0.15 each.

**Constraints:** Code change to dataset.py and train.py for curriculum masking (~50 lines). Must define difficulty ordering based on per-class AP from epoch 11. 100-epoch run. No VRAM impact. Can apply mid-training by modifying state sampling weights.

**Hypothesis:** Curriculum schedule achieves mAP50_pc=0.53-0.57 at epoch 100, with channels 16/19/22 reaching AP 0.05-0.15. Top channels (7, 9, 10) maintain AP > 0.85. Activity and pose metrics within 0.5% of baseline.

**Validation:** Implement curriculum difficulty ordering. Run 100 epochs. Compare per-class AP trajectory against uniform sampling baseline. Success criterion: channels 16, 19, 22 all >= 0.05 AP at epoch 100.

---

### Q7. SAM (Sharpness-Aware Minimization) Optimizer for Flatter Multi-Task Minima

**Context:** Current AdamW optimizer with OneCycleLR. SAM (Foret et al., ICLR 2021) finds flatter minima by minimizing the worst-case loss in a perturbation neighborhood — achieving +1-3% generalization on several benchmarks. Multi-task learning is known to require flatter minima because the optimal tradeoff surface between tasks is typically broader in flat minima (Liu et al., 2022, "Flatness-aware Multi-task Learning"). Our Kendall weighting is actively shifting task balances as log_vars converge (118 §2 Anomaly 1), meaning the model is navigating a changing tradeoff surface — especially sensitive to sharp minima that work well for one epoch's balance but poorly for the next epoch's.

**Gradient budget analysis (112-training-metrics-deep-dive.md:883-888):** At current composition (det 27.2%, pose 27.2% capped, act 14.8%, psr 30.7%), the activity head operates at the lowest gradient share. SAM's perturbation is computed on the total (Kendall-weighted) loss, meaning it naturally finds solutions where all four heads' loss landscapes are simultaneously flat — which is exactly what the activity head needs given its constrained gradient budget.

**Question:** Would replacing AdamW with SAM (rho=0.05, adaptive=True) improve combined metric from 0.363 to 0.38-0.42 by finding flatter minima that are simultaneously good for all four tasks — and would the activity head (currently gradient-starved at 14.8% share) benefit disproportionately (+0.02-0.04 macro-F1) because flatter minima require less precise gradient direction to achieve good validation performance?

**Why this matters:** SAM's 2x training cost (forward + backward for perturbation, then forward + backward for update) means 50 epochs = 100 compute epochs. This requires a careful cost-benefit analysis. If SAM provides +0.02-0.04 combined, the paper's efficiency thesis is weakened (2x training cost) but the final metric is stronger. No prior IndustReal work uses SAM. Estimated impact: +0.02-0.04 combined.

**Constraints:** Code change to train.py to implement SAM. 2x training time (50 SAM epochs = 100 compute epochs on 5060 Ti = ~14 days). VRAM: +1.5 GB for perturbation graph. Must use rho=0.05 per SAM paper recommendation for vision tasks. Can be applied mid-training by warm-starting SAM from a checkpoint (though this is non-standard).

**Hypothesis:** SAM (rho=0.05) achieves combined metric 0.38-0.42 at epoch 50 (SAM) vs 0.36-0.38 at epoch 50 (baseline). Activity improves +0.02-0.04 macro-F1. Detection improves +0.01-0.02 mAP50. Pose improves 0.3-0.5 deg.

**Validation:** Implement SAM. Run 50 epochs (100 compute epochs due to double-pass). Compare combined metric, per-task metrics at SAM-epoch 50 vs baseline-epoch 50. Success criterion: combined >= 0.39 at SAM-epoch 50 (100 compute epochs total).

---

### Q8. Stochastic Weight Averaging (SWA) with Cyclical LR Schedule

**Context:** Current EMA (decay=0.995) maintains running average of weights during training with half-life ~138 steps (111-overview.md:42-44). SWA (Izmailov et al., 2018) averages weights over the last K epochs of a cyclical or high-constant LR schedule, producing flatter minima with 1-3% generalization improvement on several benchmarks. SWA is particularly effective for multi-task problems because flat minima in the joint loss landscape tend to have better per-task tradeoffs (Cha et al., 2021, "SWA for Multi-Task Learning"). The current cosine decay schedule means the model is spending epochs 50-100 at low LR refining a single minimum — SWA can be applied as an online averaging of the cyclic checkpoints without any additional training cost.

**The key difference between EMA and SWA:** EMA applies per-step averaging throughout training, including the high-LR early phase where weights are changing rapidly. SWA averages only the final cycle of checkpoints, capturing the low-variance region. For a 100-epoch OneCycle schedule, SWA averaging epochs 75-100 (as proposed in 117 Q34) would capture 25 checkpoints where the model has converged into a single basin — the average of these 25 is guaranteed to have lower loss curvature than any individual checkpoint.

**Question:** Would SWA (averaging checkpoints from epochs 75-100 of the completed main run, using torch.optim.swa_utils with the standard SWA procedure) improve combined metric by 0.02-0.05 over the single best checkpoint (including EMA) — with the mechanism being that SWA finds a flatter minimum in the 4-task joint loss landscape, and does the activity head benefit disproportionately because its gradient share (14.8%) means it converges to a higher-curvature region than detection (27.2%)?

**Why this matters:** SWA is free — no training cost, no architecture change, no hyperparameter tuning. It uses checkpoints that already exist (or will exist at run completion). If SWA provides 0.02-0.05 combined improvement, it is the best effort-to-impact ratio in the entire project. Estimated impact: +0.02-0.05 combined, +0.01-0.03 activity macro-F1.

**Constraints:** Implement SWA as post-hoc averaging (1 hour code). No GPU cost. Requires main run to complete epochs 75-100. Cannot be validated on current epoch-11 checkpoint (too early for SWA).

**Hypothesis:** SWA (epochs 75-100 average) achieves combined metric 0.38-0.41 vs best EMA model 0.36-0.38. Activity macro-F1 improves +0.01-0.03. Detection mAP50 improves +0.01-0.02. Performance is bounded above by the quality of checkpoints in the averaging window.

**Validation:** After main run completes, load checkpoints from epochs 75-100. Average weights. Evaluate. Compare to best single-checkpoint with EMA. Success criterion: SWA combined >= 0.38.

---

### Q9. Layer-wise Adaptive Learning Rate (LLRD) for Multi-Task Fine-Tuning

**Context:** Current training uses a single backbone LR (2.5e-5, 0.1x head LR). Layer-wise Learning Rate Decay (LLRD, also called discriminative fine-tuning in Howard & Ruder, 2018) assigns decreasing learning rates to earlier layers: LR(layer_i) = LR_base * decay^depth_i. For a ConvNeXt with 4 stages (P1-P4), standard LLRD sets decay=0.95 meaning P4 gets 1x LR, P3 gets 0.95x, P2 gets 0.90x, P1 gets 0.86x. This is standard practice in ViT fine-tuning (e.g., Beit, MAE, CLIP all use LLRD).

**The critical insight from gradient analysis (112-training-metrics-deep-dive.md:883-888):** Different tasks use different feature levels: detection relies heavily on P3-P4 features (fine spatial detail), activity relies more on P1-P2 features (semantic context), pose uses all levels. LLRD allows earlier layers (which encode more general features like edges and textures) to learn more slowly, preserving their general capabilities while later layers adapt rapidly to task-specific needs. Given that detection and activity compete for different feature depths, LLRD could reduce gradient conflict between them.

**Question:** Would LLRD (backbone decay=0.90 per stage, heads at 1x LR) improve combined metric from 0.363 to 0.38-0.42 by reducing gradient competition between early-stage features (needed by activity/pose for semantics) and late-stage features (needed by detection/PSR for spatial detail) — specifically, would the per-task optimal feature depths become more specialized under LLRD, reducing the gradient conflict measured by Q23's cosine similarity tool?

**Why this matters:** The current uniform backbone LR treats all layers identically, forcing the same learning speed for edge detectors (layer 1) and assembly state classifiers (layer 4). LLRD is a zero-cost config change with proven benefits in transfer learning. Estimated impact: +0.02-0.04 combined.

**Constraints:** Config change (add LLRD schedule to optimizer code). 50-epoch run on 5060 Ti. No VRAM impact. Decay rate must be tuned (0.90 vs 0.95 vs 0.85). Cannot be applied mid-training (optimizer state mismatch).

**Hypothesis:** LLRD (decay=0.90) achieves combined metric 0.38-0.42 at epoch 25 vs 0.36-0.38 for uniform LR. Activity improvement +0.02 macro-F1 (earlier layers benefit from slower update). Detection improvement +0.01 mAP50. Grad cosine (det vs act) measured by F12 probe shows less conflict (< -0.2) vs baseline (> -0.3).

**Validation:** Implement LLRD. Sweep decay rates (0.85, 0.90, 0.95) in 25-epoch probes on 3060. Pick best, run full 50 epochs. Compare combined metric against uniform LR baseline at equivalent epochs. Success criterion: combined >= 0.39 at epoch 25.

---

### Q10. Cosine Warmup Schedule Redesign: Multi-Cycle with Cooldown

**Context:** Current OneCycleLR has a single warmup (epochs 0-3) followed by cosine decay to near-zero by epoch 100. The LR is at peak at approximately epoch 10 and decays smoothly thereafter. This schedule assumes the optimal learning rate is highest early and monotonically decreasing — appropriate for single-task supervised learning where the model is converging to one optimum. For multi-task learning with Kendall weighting, the task balance is actively changing throughout training (lv_det descending from ~0 to -0.225, lv_act from ~0 to +0.381 across epochs 7-12 per 112-training-metrics-deep-dive.md:822-825).

**The critical observation (118 §2 Anomaly 3):** The Kendall-weighted total loss is not comparable across epochs because log_vars are moving. Training loss "spikes" (2.49 to 3.02 to 3.27 at epochs 7-8) coincided with act_log_var swinging from -0.008 to +0.205 — a period where the task balance was actively churning. A single-cycle schedule assumes the gradient direction is increasingly reliable; but during Kendall convergence, the gradient direction is actually changing as weights shift between tasks.

**Question:** Would a multi-cycle cosine schedule (2 cycles of 50 epochs: warmup epochs 0-3 to peak, cosine to 0.1x peak by epoch 50, then restart: warmup epochs 51-53 to 0.5x peak, cosine to near-zero by epoch 100) improve combined metric by 0.02-0.05 over the single-cycle baseline by allowing the model to escape task-balance configurations that were locked in during the first cycle's high-LR period — and does the second cycle at lower peak LR provide better refinement of the activity head once Kendall weights have stabilized?

**Why this matters:** The current schedule plateaus at near-zero LR for epochs 60-100, during which the model is essentially converged. If the Kendall weights are still adjusting at epoch 30-40 (118 §1: "det and act still 0.3-0.6 from equilibrium"), then the plateau phase is spent refining an outdated task balance. A second cycle allows the model to re-optimize with the equilibrium task balance. Estimated impact: +0.02-0.05 combined.

**Constraints:** Config change to LR scheduler. 100-epoch run (same total epochs). No VRAM impact. Must ensure the second cycle's peak LR (0.5x) is low enough not to destabilize.

**Hypothesis:** 2-cycle schedule achieves combined metric 0.38-0.42 vs 0.36-0.38 for single-cycle at epoch 100. Activity macro-F1 improves +0.02-0.03 because the second cycle begins when Kendall weights are more stable (act closer to equilibrium). Final combined metric is higher in cycle 2 than cycle 1's plateau.

**Validation:** Implement multi-cycle scheduler. Run 100 epochs. Compare epoch-by-epoch combined metric against single-cycle baseline. Success criterion: combined >= 0.39 at epoch 100.

---


# Category 3: Loss Function Redesign (5 Questions)

PSR F1=0.144 is 84% below SOTA (0.901) and detection mAP50=0.317 is 62% below YOLOv8m (0.838). Loss function design is the most direct lever for closing these gaps. Current losses: FocalLoss (gamma=1.5) for detection, CE for activity, MSE on unit vectors for pose, binary CE for PSR. These five questions explore specialized loss functions that could address the specific failure modes identified in 112 and 118.

---

### Q11. Transition-Aware Focal Loss (TAFL) for ASD State Sequences

**Context:** The 24 ASD states follow a natural assembly order: states 0->1->2->...->23 where each transition represents adding or removing a specific component. Current FocalLoss treats each state independently, ignoring the sequential structure (112-training-metrics-deep-dive.md:688-691). Transition-Aware Focal Loss would penalize predictions that violate assembly logic (e.g., predicting state 10 "component 5 attached" before state 5 "component 5 accessible") more heavily than physically plausible errors (state 9 vs 10 differing by 1 bit).

The assembly-constraint violation analysis (111-overview.md:175-190 per-class AP): Channel 16 (binary 11101111110, AP=0.000, 9 GT) represents a state where component 4 transitions from present to absent. Under the assembly-state binary code, this is a physically plausible state but it is a rapid transition (frame count likely small). The model never detects it because standard losses treat all 24 classes uniformly -- there is no penalty for predicting an assembly-inconsistent state like "component 6 present before component 5" vs "component 6 absent but all later components present."

**Question:** Would a Transition-Aware Focal Loss that encodes the 24x24 transition-cost matrix (cost(i->j) = Hamming(binary(i), binary(j)) * assembly_violation_penalty(i,j)) improve mAP50_pc from 0.506 to 0.55-0.60 by explicitly penalizing assembly-logic violations (predicting component X present before component X's prerequisites) -- and does the 1-2 bit confusion between channels 9-12 (116-winning-aaiml-synthesis.md:171-186) get resolved because TAFL's transition-cost matrix makes Hamming-distance-2 errors cost more than Hamming-distance-1 errors?

**Why this matters:** No existing detection loss encodes assembly-state structure. The ASD binary code provides an information-theoretically optimal distance metric between states. Incorporating this into the loss is a novel contribution that cannot be replicated by any published detection method. Paper 1's YOLOv8m uses standard losses; no IndustReal work uses state-aware losses. Estimated impact: +0.04-0.09 mAP50_pc.

**Constraints:** Code change to losses.py (~30 lines for transition cost matrix). No additional VRAM. 50-epoch run on 5060 Ti. The transition matrix must be verified against the assembly manual (not just binary code distances). Can be combined with focal loss gamma as hyperparameter.

**Hypothesis:** TAFL achieves mAP50_pc=0.55-0.60 at epoch 50. Channels in the 9-12 cluster show reduced confusion (difference between min and max in cluster reduces from 0.52 to less than 0.30). Assembly-violation errors (predicting component X before prerequisites) are reduced by 30-50%.

**Validation:** Compute 24x24 transition cost matrix from binary codes + assembly manual. Implement TAFL. Run 50 epochs. Compare per-class AP confusion matrix (specifically 9-12 cluster) against standard FocalLoss. Success criterion: mAP50_pc >= 0.55 at epoch 50.

---

### Q12. Sequence-Level Contrastive Loss for PSR Temporal Consistency

**Context:** PSR F1=0.144 uses per-frame binary classification with MonotonicDecoder applying fill-forward as post-processing (114-comparability-vs-4-papers.md:280-295). The fill-forward constraint only enforces monotonicity -- it does not improve per-frame predictions. Sequence-level contrastive loss (e.g., CPC, SimCLR adapted to sequences) would learn that a correct assembly sequence (states flowing naturally 0->1->2->3) should have higher similarity in feature space than incorrect sequences (states with spurious transitions or missing steps).

The data asymmetry problem (112-training-metrics-deep-dive.md:323-337): Component 4 has only 19.1% prevalence in the dataset. The PSR decoder sees very few examples of component 4 transitioning from absent to present. A sequence-level contrastive loss could leverage the full sequences (188K frames) even when the per-component labels are sparse -- the contrastive objective would learn that sequences where component 4 activates in the correct temporal context are "correct" even if individual frame labels are noisy.

**Question:** Would adding a sequence-level contrastive loss (InfoNCE with temperature=0.07, positive pairs = adjacent frames in same assembly state, negative pairs = frames from different assemblies or distant temporal positions) to the PSR decoder training, weighting at lambda=0.3 relative to the binary classification loss, improve PSR F1 from 0.144 to 0.25-0.40 by learning smoother and more consistent feature representations across temporal sequences -- and does the contrastive objective outperform standard temporal smoothing (Q19) because it learns which temporal transitions are physically possible rather than merely penalizing all changes?

**Why this matters:** Contrastive learning for PSR is novel -- no published PSR paper (B3, STORM-PSR, or any others) uses contrastive objectives. The 188K labeled frames provide abundant data for contrastive pairs even when per-component labels are sparse. If this succeeds, the paper introduces a new paradigm for PSR training. Estimated impact: +0.10-0.25 PSR F1.

**Constraints:** Code change to losses.py (~80 lines for contrastive loss with temporal sampling). 50-epoch run. VRAM increase ~1.5 GB due to memory bank for negative samples. Must carefully define positive/negative pair sampling strategy to avoid false negatives (different phases of same assembly state should be positive).

**Hypothesis:** Contrastive auxiliary loss (lambda=0.3) achieves PSR F1=0.25-0.40 at epoch 50 (vs 0.144). POS remains above 0.95. The improvement is concentrated on rare components (h4, h7-h10) which benefit most from the sequence-level signal. Activity and detection features also improve (+0.01 each) due to shared backbone benefiting from smoother temporal features.

**Validation:** Implement sequence-level contrastive loss. Run 50 epochs with lambda sweep {0.1, 0.3, 0.5}. Compare PSR F1, per-component F1, POS. Success criterion: PSR F1 >= 0.25.

---

### Q13. Geodesic Loss with Uncertainty-Aware Weighting for Ego-Pose

**Context:** Current ego-pose uses MSE on 3D unit vectors (112-training-metrics-deep-dive.md:721-749). Forward MAE=8.14 deg, up MAE=7.06 deg per 111-overview.md:751-757. The geodesic loss (direct angle between predicted and ground-truth rotation) is proposed in Q11 with expected improvement to 7.0-7.5 deg. However, the head pose predictions have varying uncertainty -- slow head movements produce more accurate predictions than rapid turns -- and a single loss treats all samples equally.

The variance analysis (112-training-metrics-deep-dive.md:755-761): Pose loss RMS ranges from 0.02 (stable head) to 0.15 (during rapid movement). An uncertainty-weighted geodesic loss would naturally down-weight high-variance predictions (rapid movement) and up-weight low-variance ones (stable fixation), improving overall MAE.

**Question:** Would an uncertainty-aware geodesic loss that learns per-sample variance (adding 2 learnable parameters, one for forward, one for up direction) -- computed as Loss = sum_i (geodesic(v_pred_i, v_gt_i) / sigma_i^2 + log(sigma_i^2)) -- improve forward MAE from 8.14 to 6.5-7.0 deg and up MAE from 7.06 to 5.5-6.0 deg by allowing the model to focus learning on low-uncertainty (stable) head poses while discounting high-uncertainty (rapid movement) samples -- and does the learned variance parameter provide a principled measure of prediction confidence that can be reported alongside the MAE in the paper?

**Why this matters:** Uncertainty-aware losses are standard in deep regression (Kendall & Gal, 2017) but are applied to rotation/unit-vector regression by no published head pose estimation work. The estimated HoloLens 2 sensor noise floor is 5-7 deg (116-winning-aaiml-synthesis.md:573). Achieving 6.5-7.0 deg would approach this floor and make the ego-pose claim substantially more defensible. Estimated impact: -1.0 to -1.6 deg forward MAE.

**Constraints:** Code change to losses.py (~20 lines for uncertainty wrapping of geodesic loss). 25-epoch ablation on 3060. No VRAM impact. Can be combined with position-loss removal (Q12). Must initialize sigma to 1.0 for training stability.

**Hypothesis:** Uncertainty-weighted geodesic loss achieves forward MAE=6.5-7.0 deg and up MAE=5.5-6.0 deg at epoch 25 (vs 8.14/7.06 baseline). The learned sigma is 0.8-1.2 for most samples (relative weighting 0.7-1.6x) with higher values during rapid head movements. Combined with Q12 (position removal), forward MAE reaches 6.2-6.8 deg.

**Validation:** Implement uncertainty-aware geodesic loss. Run 25-epoch ablation. Compare forward/up MAE at epochs 5, 10, 15, 20, 25. Success criterion: forward MAE <= 7.0 deg at epoch 25.

---

### Q14. PSR Cross-Entropy with Component-Order Regularization

**Context:** PSR decoder uses per-component binary classification with fill-forward monotonicity enforcement (114-comparability-vs-4-papers.md:280-295). The loss does not encode assembly order constraints. Component-order regularization would penalize the decoder when it predicts component 6 (later step) activated before component 5 (earlier step) based on frame-level features, even if the fill-forward ultimately fixes the order.

The cascading error in PSR (112-training-metrics-deep-dive.md:323-337, 1288-1300): When component h4 (19.1% prevalence) is detected late or missed, every subsequent component's detection is delayed by the fill-forward constraint. The per-component F1 shows this cascade: h4's F1 near zero pulls down h5-h10. A component-order regularization loss that penalizes h4 detection delays before other components would specifically target this cascading bottleneck.

**Question:** Would adding an order-regularization term to the PSR loss -- computed as the KL divergence between the empirical component activation order and the canonical assembly order, weighted by inverse prevalence (rare components h4, h7-h10 get 3-5x weight) -- improve PSR F1 from 0.144 to 0.22-0.35 by directly penalizing the cascading delay chain that starts when rare components are missed -- and does this loss interact positively with per-component thresholds (Q18) by making the decoder's output order more correct before thresholding is applied?

**Why this matters:** This is a novel loss for fill-forward decoders. No published PSR method uses order-regularization, because they use transition-detection rather than per-frame classification. The underlying insight -- that order violations at the per-frame level cascade -- is specific to the fill-forward paradigm and represents a methodological contribution. Estimated impact: +0.08-0.20 PSR F1.

**Constraints:** Code change to losses.py (~40 lines for order regularization). 25-epoch ablation on 3060. Must define canonical order from assembly manual. Regularization weight lambda to sweep {0.1, 0.3, 0.5, 1.0}. Can be combined with Q18 thresholds and Q19 smoothing.

**Hypothesis:** Order regularization (lambda=0.3) achieves PSR F1=0.22-0.35 at epoch 25. Component h4 F1 improves from near-zero to 0.15-0.30. The cascading delay to components h5-h10 is reduced by 2-4 frames (measured as tau reduction). POS drops slightly (0.968 to 0.94-0.96) but remains above SOTA.

**Validation:** Implement order-regularization loss. Sweep lambda. Run 25 epochs each variant. Compare PSR F1, per-component F1, tau. Success criterion: PSR F1 >= 0.22.

---

### Q15. Multi-Task NT-Xent Loss for Cross-Task Feature Alignment

**Context:** The four tasks share a backbone but have separate heads with no explicit mechanism to ensure backbone features are useful for all tasks simultaneously (111-overview.md:30-35). The gradient conflict hypothesis (117 Q23: cos(det, pose) = -0.3 to -0.5) suggests tasks compete for backbone features. NT-Xent (Normalized Temperature-Scaled Cross-Entropy Loss, from SimCLR) can be adapted to multi-task learning by treating the same backbone feature processed by different task heads as "positive pairs" and features from different batches as "negative pairs."

**Mechanism:** For a batch of N samples, each sample produces features f_i (backbone output). If features are useful for all 4 tasks, then f_i should be closer to f_i itself under different augmentations than to f_j (j != i). A multi-task NT-Xent loss computes: Loss = -log(exp(sim(f_i, f_i')/tau) / sum_j exp(sim(f_i, f_j')/tau)) where f_i' is the same sample after augmentation. This encourages the backbone to produce sample-specific features that are invariant to augmentation -- which is exactly the property needed for a backbone that serves 4 disparate tasks.

**Context from gradient composition (112-training-metrics-deep-dive.md:883-888):** If detection (27.2%) and pose (27.2%) are conflicting as Q23 hypothesizes, then the backbone is currently torn between producing sharp features (det) and smooth features (pose). NT-Xent would explicitly enforce that features are "good for all tasks" by measuring whether the sample-specific feature vector is distinct from other samples' feature vectors, independent of augmentation.

**Question:** Would adding a multi-task NT-Xent loss (temperature=0.5, weight=0.1 relative to task losses, using two augmentations of each input image as positive pairs) improve combined metric from 0.363 to 0.39-0.44 by reducing gradient conflict between tasks -- specifically, would this loss decrease the det-pose gradient cosine conflict (currently hypothesized at -0.3 to -0.5 per Q23) to near-zero by explicitly enforcing that backbone features are discriminative across samples regardless of which task head is using them?

**Why this matters:** This is a novel application of contrastive learning to multi-task gradient conflict resolution. No prior work uses NT-Xent within a multi-task backbone training framework for detection + pose + activity + PSR. Estimated impact: +0.03-0.06 combined, primarily through reduced task interference.

**Constraints:** Code change to losses.py and train.py (~60 lines for NT-Xent with multi-view augmentation). 25-epoch ablation on 3060. Requires two forward passes per batch (two augmentations). VRAM increase ~1 GB for second augmentation branch. Temperature tau to sweep {0.1, 0.5, 1.0}. Loss weight lambda to sweep {0.05, 0.1, 0.2}.

**Hypothesis:** Multi-task NT-Xent (tau=0.5, lambda=0.1) achieves combined metric 0.39-0.44 at epoch 25. Det-pose gradient cosine (measured by F12 probe) improves from -0.3 to -0.5 (baseline) to > -0.1. All four tasks show improvement, with the largest gain on activity (currently most gradient-starved at 14.8%).

**Validation:** Implement multi-task NT-Xent. Run 25-epoch ablation with hyperparameter sweep. Compare combined metric, per-task metrics, grad cosine probe. Success criterion: combined >= 0.39 at epoch 25.

---

# Category 4: Data Strategy Changes (5 Questions)

Current data: 26K train, 38K val, 188K labeled frames with 4,710 detection-GT-bearing frames. Only 17.89% of training frames carry detection boxes. These five questions explore synthetic data, pseudo-labeling, sampling strategies, and data augmentation.

---

### Q16. Hard Example Mining with Temporal Tubelets from Assembly Videos

**Context:** The current 26K training frames are uniformly sampled from the 188K labeled pool (111-overview.md:135-137). Three channels have AP=0.000 (16, 19) and channel 22 has 0.063 -- all transitional states appearing for only 1-3 frames during assembly state changes. Standard uniform sampling captures very few examples of these rapid transitions because they represent a tiny fraction of total frames.

The temporal tubelet concept: Instead of uniform frame sampling, sample temporal tubelets (short sequences of 5-10 consecutive frames) with a bias toward frames where the assembly state is changing. This ensures that transitional states (channels 16-23, which have 5-28 GT instances each) are adequately represented in the training set. The tubelet sampling probability can be proportional to the frame's state-change score (derived from the binary code delta between consecutive frames).

**Question:** Would temporally biased tubelet sampling (5-frame tubelets, sampling probability proportional to binary code delta between consecutive frames) increase effective training examples for channels 16/19/22 by 3-10x (from 5-28 to 50-200), improving their AP from near-zero to 0.05-0.15, while maintaining top-channel AP through uniform tubelet sampling of non-transitional frames?

**Why this matters:** The current sampling strategy is uniform and misses the temporal structure of assembly. By explicitly oversampling transitions, we address the root cause of rare-state underperformance. This does not require generating new data -- it requires smarter sampling from the existing 188K frames. Estimated impact: +0.03-0.06 mAP50_pc, concentrated on transitional channels.

**Constraints:** Code change to dataset.py (~40 lines for tubelet sampling with transition bias). No additional VRAM. 50-epoch run on 5060 Ti. Must ensure class balance is monitored (tubelet bias could over-represent certain transition types). The transition bias strength alpha to sweep {1, 2, 5}.

**Hypothesis:** Tubelet sampling (alpha=2) achieves mAP50_pc=0.53-0.57 at epoch 50. Channels 16, 19, 22 reach AP 0.05-0.15 (vs 0.000, 0.000, 0.063). Top channels (7, 9, 10) stay within 0.01 of baseline. PSR F1 improves +0.02-0.05 because the model sees more transition examples during training.

**Validation:** Implement transition-biased tubelet sampling. Run 50 epochs with alpha sweep. Compare per-class AP with uniform baseline. Success criterion: channels 16, 19, 22 all >= 0.05 AP.

---

### Q17. Semi-Supervised Detection with FixMatch on 188K Unlabeled Frames

**Context:** Only 4,710 of 26K training frames have detection GT (111-overview.md:139). The remaining 82% of training frames plus 150K additional labeled frames (188K total labeled frames in dataset, of which only 26K are used for training) have activity and pose labels but NOT detection labels. FixMatch (Sohn et al., NeurIPS 2020) combines consistency regularization (weak augmentation produces pseudo-labels, strong augmentation enforces consistency) and has demonstrated 3-5% AP improvement on COCO with 10% labeled data.

The asymmetry in our dataset: The 188K frames have activity labels (69 classes) and pose labels (head orientation). A FixMatch-style approach could use the activity- and pose-labeled frames as the "unlabeled" pool for detection, generating pseudo-detection-labels on these frames using consistency between weak and strong augmentations. This leverages the existing 150K+ frames that currently contribute zero detection supervision.

**Question:** Would semi-supervised FixMatch detection training -- using the 4,710 detection-GT frames as the labeled set and the remaining ~184K activity/pose-labeled frames as the unlabeled set, with weak augmentation (flip + small crop) for pseudo-label generation and strong augmentation (RandAugment, 2 ops, magnitude 9) for consistency -- improve mAP50 from 0.317 to 0.40-0.48 and mAP50_pc from 0.506 to 0.58-0.65 by providing effective detection supervision on frames that currently contribute zero positive detection gradient?

**Why this matters:** This is the single highest-impact data strategy available because it exploits the unique multi-label structure of the dataset (activity + pose labels on every frame, detection labels on a subset). No existing IndustReal method uses semi-supervised detection training. Paper 1's YOLOv8m is fully supervised (synthetic + real GT). If FixMatch works, the paper demonstrates a novel benefit of multi-task data collection. Estimated impact: +0.08-0.16 mAP50, +0.07-0.14 mAP50_pc.

**Constraints:** Code change to train.py (~150 lines for FixMatch loop with weak/strong augmentation streams). 50-epoch run on 5060 Ti. Training cost increases ~2x due to dual augmentation forward passes + consistency computation. Must carefully tune confidence threshold for pseudo-label acceptance (standard is 0.7-0.9). Strong augmentation pipeline must be compatible with activity/pose labels (geometric consistency).

**Hypothesis:** FixMatch (confidence threshold=0.9, strong aug = RandAugment 2/9) achieves mAP50=0.40-0.48 at epoch 50 (vs 0.317 at epoch 11 baseline). mAP50_pc=0.58-0.65. The improvement is largest on channels with sufficient GT anchors (channels 7, 9, 10 with 50+ GT instances show 0.05-0.10 AP gain). Rare channels benefit less because the pseudo-label quality on frames without GT is poor for rare classes.

**Validation:** Implement FixMatch training loop. Run 50 epochs. Compare mAP50, mAP50_pc, per-class AP against supervised-only baseline. Success criterion: mAP50 >= 0.40 at epoch 50.

---

### Q18. Counterfactual Data Augmentation via Assembly State Manipulation

**Context:** The ASD taxonomy has 24 states with a known binary code (111 Section 3). The natural dataset only contains states that occur during actual assembly. Many binary code combinations do not appear. For example, a state with "component 3 attached but component 2 not attached" may be physically impossible in the assembly sequence -- but the model does not know this constraint. Training on counterfactual examples (generated by digitally composing/dissembling components in images) would teach the model the assembly-state grammar.

The core idea: For each training image with a known ASD state (binary code), generate counterfactual examples by digitally removing specific components (using the known 2D bounding box locations from GT), creating states that are physically consistent but rarely observed. For example, from a state 10 image (components 0-5 present, 6-23 absent), generate state 5 by removing components 2-5 via inpainting. This produces training examples for intermediate states that are underrepresented in the natural data.

**Question:** Would counterfactual data augmentation -- generating 50,000 synthetic training images by digitally manipulating component visibility in existing GT images (removing components via inpainting, adding components via composite from other images), targeting the 10 most underrepresented states (channels 13-23 with fewer than 50 GT instances each) -- improve mAP50_pc from 0.506 to 0.56-0.62 by providing 3-10x more training examples for rare states, and does this approach outperform synthetic data from Unity (Q37) because the image statistics are perfectly matched to the real data distribution?

**Why this matters:** Counterfactual augmentation is a novel approach specific to assembly-state detection where the state decomposition into independent components is known. Unlike generic synthetic data (Q37), counterfactual augmentation uses real image backgrounds + real component textures, minimizing domain gap. Estimated impact: +0.05-0.11 mAP50_pc.

**Constraints:** Code change to dataset.py (~120 lines for counterfactual pipeline using GT boxes + inpainting). Generation cost: ~10 hours CPU for 50K images (offline). No VRAM impact during training. Must ensure counterfactual states are physically valid (respect assembly constraints). Inpainting quality matters -- poor inpainting introduces artifacts that could confuse the model.

**Hypothesis:** Counterfactual augmentation achieves mAP50_pc=0.56-0.62 at epoch 50. Channels 16/19/22 show AP improvement of 0.08-0.20 each because they receive 5-10x more training examples. Top channels (7, 9, 10) maintain AP > 0.85. The improvement generalizes to real validation data because the counterfactual examples use real image statistics.

**Validation:** Generate 50K counterfactual images for rare states. Add to training set. Run 50 epochs. Compare per-class AP for target states against baseline and against synthetic data augmentation (if Q37 runs). Success criterion: mAP50_pc >= 0.56 at epoch 50.

---

### Q19. Curriculum Sampling Based on Per-Class Learning Progress

**Context:** Per-class AP ranges from 0.000 (channels 16, 19) to 0.886 (channel 9) at epoch 11 (111-overview.md:175-190). Some classes are learned quickly while others plateau early. Learning-based curriculum sampling (MentorNet, Jiang et al., 2018) adjusts each class's sampling probability based on its validation performance -- focusing training on classes that are still improving while reducing sampling of saturated classes.

The current sampling problem: In a uniform batch, channels with high AP (7, 9, 10) dominate gradient because they produce more positive samples. Channels with zero AP (16, 19) produce almost no positive samples and their gradient is entirely from the FocalLoss on negative predictions. Curriculum sampling would increase the batch ratio of underperforming classes until their AP starts rising.

**Context from OHEM analysis (117 Q2):** The effective OHEM ratio for frames with 1-5 GT boxes is 6.4:1 to 32:1 (dominated by min_neg=32 floor). Curriculum sampling that ensures rare classes appear in more batches directly addresses the OHEM imbalance by giving rare classes more opportunities to generate positive anchors.

**Question:** Would learning-progress-based curriculum sampling (each class's sampling weight updated every 5 epochs: w_c = max(1, target_AP - current_AP_c) * base_weight, where target_AP is the running max observed for that class) improve mAP50_pc from 0.506 to 0.53-0.57 within 50 epochs by concentrating sampling on classes that are still learning rather than classes that have plateaued -- and does this approach outperform fixed oversampling (Q16's tubelets) because it adaptively reduces sampling once a class reaches acceptable performance, avoiding overfitting on rare classes with limited unique examples?

**Why this matters:** Adaptive curriculum based on per-class AP is novel for assembly state detection. No prior work uses online learning progress to adjust ASD class sampling. This requires only sampling weight changes -- no architecture or loss changes. Estimated impact: +0.02-0.05 mAP50_pc.

**Constraints:** Code change to dataset.py (~30 lines for learning-progress tracking + sampling weight update). 50-epoch run on 5060 Ti. No VRAM impact. Must tune base_weight and update frequency (every 5 epochs is standard).

**Hypothesis:** Adaptive curriculum achieves mAP50_pc=0.53-0.57 at epoch 50. Channels 16/19/22 reach AP 0.05-0.10 (progress-based sampling increases their presentation rate 3-5x). Top channels (7, 9, 10) maintain AP within 0.01 of baseline (their weight decreases as they plateau).

**Validation:** Implement learning-progress curriculum sampling. Run 50 epochs. Compare per-class AP trajectory against uniform sampling. Success criterion: mAP50_pc >= 0.53 at epoch 50.

---

### Q20. Multi-Modal Augmentation: Optical Flow as Temporal Supervisory Signal

**Context:** Current training uses static RGB frames only. Optical flow between consecutive frames captures motion information: a tightening hand moves in a specific circular pattern, a screwdriver advances linearly. Flow provides a free supervisory signal because it can be computed from any pair of consecutive frames (no annotation required) and correlates with both activity class (what action is being performed) and assembly state (what is happening to the components).

The flow-activity correlation: Assembly activities have distinctive motion signatures: "tighten" produces rotational flow around the screw axis, "insert" produces translational flow toward the assembly, "check" produces minimal flow (head turns). The current per-frame activity macro-F1=0.110 could benefit significantly from motion features because static frames are highly ambiguous (a hand near a screw could be about to tighten, about to loosen, or just resting).

**Question:** Would training with auxiliary optical flow prediction (RAFT-style flow loss on backbone features, weight=0.1, using FlowNet2 compute RGB flow as pseudo-targets offline) improve act_macro_F1 from 0.110 to 0.15-0.20 and det_mAP50 by 0.01-0.03 by forcing the backbone to learn motion-sensitive features -- and does the activity head benefit more (+0.04-0.09 macro-F1) than detection (+0.01-0.03 mAP50) because motion is more discriminative for actions than for static object states?

**Why this matters:** Optical flow is free (requires no labels, only consecutive frames). No IndustReal paper uses flow-based multi-task training. If flow provides activity improvement comparable to a temporal head (Q7 estimated +0.03-0.05) without requiring temporal inference architecture, the efficiency thesis strengthens significantly. Estimated impact: +0.04-0.09 act macro-F1, +0.01-0.03 det mAP50.

**Constraints:** Code change to model.py for flow prediction head (~50 lines) + losses.py for flow loss. Requires offline flow computation for all 188K frames (~2 days GPU preprocessing, or use GPU-efficient FlowNet2-S). Adds ~5M flow head parameters. VRAM increase ~2 GB during training with RAFT forward pass. Must decide flow input representation (2-channel warp vs 3-channel color-coded).

**Hypothesis:** Auxiliary flow prediction achieves act_macro_F1=0.15-0.20 (vs 0.110) and det_mAP50=0.33-0.35 (vs 0.317) at epoch 50. Motion-dominant actions (tighten, loosen, insert) show largest activity gains. The flow head's auxiliary loss stabilizes at 0.2-0.3 after 20 epochs.

**Validation:** Compute optical flow offline for training set. Add flow prediction auxiliary head. Run 50 epochs. Compare act_macro_F1, det_mAP50 against RGB-only baseline. Success criterion: act_macro_F1 >= 0.15 at epoch 50.

---

# Category 5: Multi-Task Balancing (5 Questions)

Kendall uncertainty weighting with HP_PREC_CAP is current method. Gradient composition: det 27.2%, pose 27.2% (capped), act 14.8%, psr 30.7% (112-training-metrics-deep-dive.md:883-888). These five questions explore alternative balancing algorithms (GradNorm, PCGrad, CAGrad, DWA, IMTL-G) and their potential to improve the gradient-starved activity head.

---

### Q21. CAGrad (Conflict-Averse Gradient Descent) for 4-Task Balancing

**Context:** Current Kendall uncertainty weighting requires all four task gradients to be combined via weighted sum. When task gradients conflict (as hypothesized for detection vs pose per 117 Q23 with cos estimated at -0.3 to -0.5), the weighted sum produces a gradient that helps neither task optimally. CAGrad (Liu et al., NeurIPS 2021) explicitly optimizes for gradient direction that improves all tasks simultaneously by solving a small optimization problem at each step: find gradient g that maximizes the minimum task improvement rate.

The key insight for our gradient composition (112-training-metrics-deep-dive.md:883-888): Detection (27.2%) and PSR (30.7%) dominate gradient share while activity struggles at 14.8%. CAGrad will naturally find gradient directions that prevent the dominant tasks from suppressing the minority task. Unlike PCGrad (which projects conflicting gradients to remove conflict) or GradNorm (which equalizes gradient magnitudes), CAGrad explicitly optimizes for the worst-case task improvement rate -- directly addressing the activity head chronic underperformance.

**Question:** Would replacing Kendall uncertainty weighting with CAGrad (with convergence_alpha=0.5, the standard hyperparameter controlling tradeoff between average and worst-case improvement) increase act_macro_F1 from 0.110 to 0.15-0.20 while maintaining detection within 0.01 of 0.317 -- specifically, would CAGrad worst-case-aware optimization disproportionately benefit the gradient-starved activity head (14.8% current share) by finding gradient directions that improve activity without reducing detection or PSR performance?

**Why this matters:** CAGrad has demonstrated superior multi-task performance on NYUv2 (semantic segmentation + depth + surface normal) where task gradients naturally conflict. Our setup (detection + pose + activity + PSR) has similar gradient conflict patterns. No prior IndustReal work uses CAGrad. This is a config-level change (install CAGrad library) with no architecture modifications. Estimated impact: +0.04-0.09 act macro-F1, +0.01-0.02 combined.

**Constraints:** Code change to train.py (~30 lines to replace Kendall weighted sum with CAGrad). 25-epoch ablation on 3060. Small computational overhead (~0.5 ms per step for the CAGrad inner optimization). Must sweep alpha {0.4, 0.5, 0.6}. Must verify CAGrad does not destabilize log_var-based uncertainty estimation (if kept as auxiliary loss).

**Hypothesis:** CAGrad (alpha=0.5) achieves act_macro_F1=0.15-0.20, det_mAP50=0.31-0.33 (within 0.01 of baseline), pose MAE=8.0-8.5 (within 0.5 deg), PSR F1=0.13-0.17 (within 0.03). The combined metric improves to 0.37-0.40 (vs 0.363 baseline). Activity effective gradient share rises from 14.8% to 20-25%.

**Validation:** Implement CAGrad loss weighting. Run 25-epoch ablation with alpha sweep. Compare all 4 task metrics against Kendall baseline at equivalent epochs. Success criterion: combined >= 0.38 at epoch 25.

---

### Q22. GradNorm with Adaptive Alpha for Gradient Equilibrium

**Context:** GradNorm (Chen et al., ICML 2018) balances multi-task learning by equalizing gradient magnitudes across tasks, with an alpha parameter controlling the strength of balancing. The current hypothesis (117 Q22) predicts activity -0.03 gain at detection -0.02 cost -- but this assumes fixed alpha. Adaptive alpha -- where alpha increases when a task loss plateau is detected and decreases when it is still improving -- could achieve better balance.

Current gradient magnitudes (112-training-metrics-deep-dive.md:883-888): Raw gradient norms: det=27.2% of total, pose=27.2%(capped), act=14.8%, psr=30.7%. The ideal for 4 balanced tasks would be 25% each. Activity 14.8% is 40% below balanced, PSR 30.7% is 23% above. GradNorm would adjust weights so all four tasks contribute approximately equal gradient magnitudes to the backbone.

The critical limitation of Kendall weighting in our setup: Kendall learns a per-task precision parameter that weights the loss. But it does not directly control gradient magnitudes -- tasks with inherently larger losses (like detection with its multi-box regression) naturally dominate regardless of precision. GradNorm operates on gradients directly, which is a more direct intervention for the gradient imbalance we observe.

**Question:** Would GradNorm with adaptive alpha (alpha starts at 1.0 and decays to 0.12 over 25 epochs, tracking the rate of task-specific loss convergence) increase activity gradient share from 14.8% to 20-25% and improve act_macro_F1 from 0.110 to 0.14-0.18, while maintaining detection within 0.02 of 0.317 -- and does the adaptive alpha mechanism outperform fixed alpha by preventing over-regularization of PSR (which needs higher gradient early for component detection, but less later)?

**Why this matters:** GradNorm is a well-established multi-task balancing method (ICML 2018, 1500+ citations) that operates on gradients directly. Its application to detection + pose + activity + PSR on the IndustReal dataset is novel. The adaptive alpha variant is a potential methodological contribution to the multi-task literature. Estimated impact: +0.03-0.07 act macro-F1, combined flat to +0.02.

**Constraints:** Code change to train.py (~50 lines for GradNorm with adaptive alpha). 25-epoch ablation on 3060. Must compute per-task gradient norms (requires additional backward hooks). VRAM increase ~500 MB for gradient storage. Must tune initial alpha and decay schedule.

**Hypothesis:** GradNorm (adaptive alpha) achieves act_macro_F1=0.14-0.18, det_mAP50=0.30-0.32, PSR F1=0.12-0.16 at epoch 25. Activity gradient share rises from 14.8% to 20-25%. The Kendall log_var parameters (if kept) move closer together (spread reduces from current 1.2 range to less than 0.8).

**Validation:** Implement GradNorm with adaptive alpha. Run 25-epoch ablation. Compare per-task metrics, gradient composition against Kendall baseline. Success criterion: act_macro_F1 >= 0.14 with det_mAP50 >= 0.30.

---

### Q23. IMTL-G (Impartial Multi-Task Learning with Gradient Normalization)

**Context:** IMTL-G (Liu et al., CVPR 2022) proposes that optimal multi-task learning should give each task equal influence on the shared parameters update direction. It normalizes task gradients by their L2 norm and ensures the combined gradient has equal projection onto each task gradient direction. This is mathematically simpler than GradNorm or CAGrad while achieving comparable or better results on standard benchmarks.

**Mathematical advantage for our gradient conflict scenario:** When detection (27.2%) and pose (27.2% capped) gradients are in conflict (cos less than 0), the combined gradient in standard methods points in a direction that helps neither task optimally. IMTL-G ensures each task contributes equally to the final gradient direction, which naturally handles gradient conflicts by preventing the dominant task from determining the direction. For our 4-task setup where detection and PSR dominate direction, IMTL-G would ensure activity has equal directional say.

**Question:** Would IMTL-G gradient balancing improve activity macro-F1 from 0.110 to 0.13-0.17 while maintaining detection and PSR within 0.01 of baseline -- and does IMTL-G equal-direction property specifically help the activity head (currently 14.8% gradient share) by giving it equal influence on the backbone update direction even while its gradient magnitude remains small?

**Why this matters:** IMTL-G is elegant (fewer hyperparameters than GradNorm or CAGrad) and directly addresses the gradient conflict problem. If it works for our 4-task setup, it is a clean methodological contribution requiring minimal implementation effort. Estimated impact: +0.02-0.06 act macro-F1.

**Constraints:** Code change to train.py (~20 lines for IMTL-G normalization). 25-epoch ablation on 3060. No VRAM impact beyond storing per-task gradients. Must handle the HP_PREC_CAP interaction (capped pose gradient should be normalized differently -- may need to decouple pose from the IMTL-G computation).

**Hypothesis:** IMTL-G achieves act_macro_F1=0.13-0.17, det_mAP50=0.31-0.33, pose MAE=8.0-8.5, PSR F1=0.13-0.17 at epoch 25. Activity directional influence rises from its current de facto zero (14.8% share diluted by dominant tasks) to 25% (equal to every other task). The combined metric improves to 0.37-0.40.

**Validation:** Implement IMTL-G. Run 25-epoch ablation. Compare all 4 task metrics against Kendall baseline. Success criterion: combined >= 0.37 with activity >= 0.13.

---

### Q24. Dynamic Weight Averaging (DWA) with Task-Specific Temperature

**Context:** DWA (Liu et al., CVPR 2019) computes task weights based on the rate of loss change: tasks that are improving quickly get lower weight (they need less help), tasks that are improving slowly get higher weight (they need more gradient). This is conceptually complementary to Kendall uncertainty weighting -- Kendall measures prediction uncertainty, DWA measures learning progress.

The learning rate asymmetry (112-training-metrics-deep-dive.md:822-825, 896-903): Log_var gradients are decreasing (from 0.5 to 0.3), indicating Kendall is reaching equilibrium. But the per-task loss trajectories show different convergence rates: detection loss is still declining slowly, activity loss plateaued then recovered, PSR loss is flat. DWA would detect that activity is the slowest-improving task and increase its weight accordingly.

**Question:** Would replacing Kendall weighting with DWA (temperature T=2.0, softmax over loss ratio, updated every 5 epochs) using task-specific temperatures (higher T for activity to allow larger weight swings, lower T for detection to keep weight stable) improve act_macro_F1 from 0.110 to 0.14-0.18 by explicitly up-weighting the slowest-improving task, while maintaining detection within 0.015 of 0.317 because DWA softer weighting (vs GradNorm hard equalization) prevents catastrophic unbalancing?

**Why this matters:** DWA is simpler than GradNorm (no gradient computation needed, uses only scalar losses) and is less likely to destabilize training. Its application to our 4-task setup with known convergence rate asymmetry is natural. Estimated impact: +0.03-0.07 act macro-F1.

**Constraints:** Code change to train.py (~15 lines for DWA computation). 25-epoch ablation on 3060. No VRAM impact. Must tune temperature T for each task {1.0, 2.0, 4.0}. Update frequency to sweep {1, 5, 10} epochs.

**Hypothesis:** DWA (T_act=4.0, T_det=1.0, T_pose=2.0, T_psr=2.0) achieves act_macro_F1=0.14-0.18, det_mAP50=0.30-0.33 at epoch 25. Activity weight rises from 14.8% effective to 20-25%. The DWA loss-rate ratio for activity shows it is 2-3x slower-improving than detection, justifying the higher weight.

**Validation:** Implement DWA with task-specific temperatures. Run 25-epoch ablation. Compare all metrics against Kendall baseline. Success criterion: combined >= 0.37 with activity >= 0.14.

---

### Q25. Gradient Surgery with Gradient Vaccine (GradVac)

**Context:** Gradient Vaccine (Wang et al., ICLR 2021, "Gradient Vaccine: Investigating and Improving Multi-Task Optimization in Massively Multilingual Models") introduces a controlled gradient projection mechanism: instead of removing conflict entirely (PCGrad) or optimizing for worst-case (CAGrad), GradVac maintains an exponential moving average of gradient cosine similarities and gently pushes task gradients toward positive similarity. This prevents dominant tasks from suppressing minority tasks while still allowing beneficial task interactions.

The difference from PCGrad and CAGrad: PCGrad removes conflicting components entirely (which can discard useful gradient signal). CAGrad optimizes for worst-case improvement (which can be conservative). GradVac adaptively adjusts gradient direction based on historical similarity, which is gentler and more appropriate for our setup where task relationships may be task-dependent (some tasks have high conflict, others low).

Context from epoch-11 gradient analysis (112-training-metrics-deep-dive.md:883-888): If cos(det, pose) = -0.3 to -0.5 (as hypothesized in Q23), this is moderate conflict -- not extreme (-0.9) and not aligned (+0.9). GradVac is specifically designed for this regime: it would gently push det and pose gradients toward zero conflict without the hard projection that PCGrad applies.

**Question:** Would GradVac gradient surgery (with default EMA beta=0.99, target similarity=0.2) improve combined metric from 0.363 to 0.38-0.42 by reducing harmful gradient conflict between detection and pose (currently estimated at cos=-0.3 to -0.5) while preserving beneficial gradient alignment between detection and PSR (likely positive cos because PSR depends on detection features) -- and does GradVac adaptive conflict resolution outperform PCGrad hard projection by maintaining beneficial multi-task interactions?

**Why this matters:** GradVac is the most principled gradient surgery method for moderate gradient conflicts, which is exactly our regime. It adds minimal computational overhead (EMA of cosine similarities). No prior IndustReal work uses gradient surgery. Estimated impact: +0.02-0.05 combined.

**Constraints:** Code change to train.py (~40 lines for GradVac gradient surgery). 25-epoch ablation on 3060. Must pre-compute gradient similarities for all 6 task pairs (det-pose, det-act, det-psr, pose-act, pose-psr, act-psr). VRAM increase ~200 MB for gradient storage. Target similarity to sweep {-0.1, 0.0, 0.2, 0.4}.

**Hypothesis:** GradVac (target_sim=0.2) achieves combined metric 0.38-0.42 at epoch 25. Det-pose cosine improves from -0.3 to -0.5 to greater than -0.1. Per-task metrics: det_mAP50 improves to 0.33-0.35 (conflict reduction helps detection), act_macro_F1 improves to 0.13-0.16 (reduced det dominance), pose MAE stays within 0.3 deg. PSR F1 stays within 0.02 of baseline.

**Validation:** Implement GradVac. Run 25-epoch ablation with target similarity sweep. Compare per-task metrics, gradient cosine similarities against baseline (no surgery). Success criterion: combined >= 0.38 at epoch 25.

---

# Category 6: Cross-Dataset Transfer (5 Questions)

The IKEA ASM dataset provides an external benchmark for assembly state detection and activity recognition. These five questions explore how to maximize transfer from IKEA ASM pre-training or co-training, and what architectural choices improve generalization across differently-structured assembly tasks.

---

### Q26. IKEA ASM Pre-Training for Assembly Feature Learning

**Context:** IKEA ASM (Ben-Shabat et al., CVPR 2021) provides 12 furniture assembly sequences with 24K frames, multi-view RGB-D, and assembly state annotations. While the furniture types differ from IndustReal bolt-and-washer assembly, the underlying visual primitives (object detection, spatial relationships, assembly state transitions) are shared. Pre-training on IKEA ASM before fine-tuning on IndustReal would provide the backbone with assembly-specific visual knowledge that ImageNet does not provide (assembly tools, partial assemblies, human hands manipulating objects).

The transfer learning hypothesis: ConvNeXt-Tiny trained on ImageNet (no assembly knowledge) must learn all assembly-specific features from scratch. IKEA ASM pre-training would teach the backbone: (1) what a partial assembly looks like, (2) how spatial relationships between parts determine assembly state, (3) how human hands interact with components. This domain-specific pre-training could be more valuable than ImageNet because it directly teaches the relevant visual concepts.

**Question:** Would IKEA ASM pre-training (50 epochs on IKEA ASM multi-view RGB-D assembly state detection, then fine-tuning 50 epochs on IndustReal) improve det_mAP50 from 0.317 to 0.38-0.44 and act_macro_F1 from 0.110 to 0.14-0.18 compared to the random-init baseline -- and does the IKEA ASM pre-training benefit transfer more to detection (+0.06-0.12 mAP50) than to activity (+0.03-0.07 macro-F1) because assembly state detection shares more visual structure across datasets than activity recognition?

**Why this matters:** Cross-dataset transfer for assembly state detection is unexplored. IKEA ASM is the only large-scale assembly state dataset. Demonstrating successful transfer would be a significant result showing that assembly visual features generalize across products. No existing IndustReal paper uses IKEA ASM pre-training. Estimated impact: +0.06-0.12 mAP50, +0.03-0.07 act macro-F1.

**Constraints:** Must download and preprocess IKEA ASM (~50GB, 1 day). Code changes for dataset integration. 50-epoch pre-training + 50-epoch fine-tuning on 5060 Ti = ~14 days total. Must handle different class definitions (IKEA ASM uses different state taxonomy). Must decide which IKEA ASM data to use (RGB-only vs RGB-D, single-view vs multi-view).

**Hypothesis:** IKEA ASM pre-trained model achieves IndustReal det_mAP50=0.38-0.44 at fine-tuning epoch 25 (vs 0.317 at epoch 11 for random init). Act_macro_F1=0.14-0.18. Pre-trained features are most beneficial for low-GT channels (16, 19, 22 show +0.08-0.15 AP). The pre-training benefit decays by epoch 50 (fine-tuning reduces gap to 0.03-0.05 mAP50).

**Validation:** Train on IKEA ASM assembly state detection for 50 epochs. Fine-tune on IndustReal for 50 epochs. Compare epoch-by-epoch metrics against random-init baseline. Success criterion: det_mAP50 >= 0.38 at fine-tuning epoch 25.

---

### Q27. Multi-Dataset Training: Joint IndustReal + IKEA ASM with Task-Specific Heads

**Context:** Rather than pre-train and fine-tune, joint training on both datasets sharing a backbone but using task-specific heads could provide regularization and improve generalization. The two datasets have different assembly taxonomies, camera viewpoints, and lighting conditions -- forcing the backbone to learn viewpoint-invariant and lighting-invariant assembly features.

The regularization hypothesis: Joint training on two different assembly datasets acts as a strong regularizer, preventing overfitting to IndustReal-specific visual patterns (fixed camera angle, specific workshop lighting). The backbone must learn features that work for both IKEA furniture (different objects, different room lighting) and IndustReal components (industrial setting). This is analogous to multi-dataset training in semantic segmentation (Mapillary + Cityscapes) where joint training improves both datasets performance.

Context from current overfitting risk (118 Section 1, 7.21): With 1,769:1 parameter-to-sample ratio (46.5M params / 26K frames), overfitting risk is moderate. Joint training on IKEA ASM provides 24K additional frames (50% more data) from a different domain, directly reducing overfitting risk.

**Question:** Would joint training on IndustReal (26K frames, 4 tasks) + IKEA ASM (24K frames, assembly state + action detection), using a shared ConvNeXt-Tiny backbone with dataset-specific heads (IndustReal heads unchanged, IKEA ASM heads for its 12-class assembly state + action detection), improve IndustReal det_mAP50 from 0.317 to 0.35-0.40 and act_macro_F1 from 0.110 to 0.14-0.18 by providing 92% more training data from a complementary domain -- and does the joint training regularize the backbone sufficiently to eliminate the need for EMA (current 0.995) or other regularization?

**Why this matters:** Multi-dataset training for assembly tasks is novel. The shared backbone learns dataset-invariant assembly features -- a concept that could be extended to arbitrary assembly tasks. No prior work jointly trains on IKEA ASM and IndustReal. Estimated impact: +0.03-0.08 mAP50, +0.03-0.07 act macro-F1.

**Constraints:** Code changes for multi-dataset dataloader (~100 lines). 50-epoch joint training on 5060 Ti (~10 days). Must handle batch composition (alternate between datasets or mixed batches). Must handle different class taxonomies. IKEA ASM detection labels must be integrated with our evaluation pipeline.

**Hypothesis:** Joint training achieves IndustReal det_mAP50=0.35-0.40 and act_macro_F1=0.14-0.18 at epoch 50. IKEA ASM performance also exceeds single-dataset training (+2-3% on its metrics). The backbone feature visualization shows less dataset-specific pattern (lower domain confusion measured by a domain classifier).

**Validation:** Implement multi-dataset training loop. Run 50 epochs. Compare IndustReal metrics against single-dataset baseline. Also evaluate on IKEA ASM to measure cross-dataset performance. Success criterion: IndustReal det_mAP50 >= 0.35 at epoch 50.

---

### Q28. Domain Adversarial Training for Viewpoint-Invariant Assembly Features

**Context:** The IndustReal dataset has a fixed camera viewpoint (overhead, 60-degree angle) per 111-overview.md:90-95. The model may learn viewpoint-specific features that do not generalize. Domain adversarial training (Ganin et al., JMLR 2016) adds a domain classifier that tries to predict the camera viewpoint from backbone features, while the backbone is trained to fool this classifier -- forcing the backbone to learn viewpoint-invariant assembly features.

The viewpoint overfitting evidence (112-training-metrics-deep-dive.md:497-514, per-class AP): Channels with highest AP (7, 9, 10) correspond to states where the distinguishing components are clearly visible from the fixed overhead viewpoint. Channels with zero AP (16, 19) may correspond to states where the distinguishing component is occluded from this specific viewpoint. A model that learns viewpoint-invariant features would recognize the assembly state regardless of the viewpoint occluding effects.

**Question:** Would adding a domain adversarial branch (gradient reversal layer, 3-layer MLP domain classifier predicting camera viewpoint index from pooled backbone features, gradient reversal weight lambda=0.1) improve per-class AP on occlusion-sensitive channels (16, 19, 22) by 0.05-0.15 each by forcing the backbone to learn viewpoint-invariant assembly features that do not rely on spurious correlations with camera angle -- and does this benefit extend to other viewpoint variations (different HoloLens wearer heights, different head positions)?

**Why this matters:** Adversarial domain adaptation for viewpoint invariance in assembly tasks is novel. The fixed IndustReal camera setup means the model almost certainly has viewpoint-specific features. If adversarial training recovers the zero-AP channels, it suggests their failure is due to viewpoint-occlusion rather than insufficient training data. Estimated impact: +0.03-0.08 mAP50_pc, concentrated on transitional channels.

**Constraints:** Code change to model.py (~50 lines for domain classifier + GRL). 25-epoch ablation on 3060. VRAM increase ~300 MB. Must define viewpoint domains: by participant (12 participants, each with slightly different HL2 mounting), or by recording session (varying backgrounds). Gradient reversal weight lambda to sweep {0.01, 0.1, 0.5}.

**Hypothesis:** Domain adversarial training (lambda=0.1) improves channels 16, 19, 22 by +0.05-0.15 AP each at epoch 25. Top channels (7, 9, 10) stay within 0.01. The domain classifier accuracy drops from 80%+ to 50-60% (backbone successfully fools it). The improvement generalizes because the backbone now uses assembly features rather than viewpoint features.

**Validation:** Implement domain adversarial branch. Run 25-epoch ablation. Compare per-class AP, domain classifier accuracy against baseline. Success criterion: channels 16, 19, 22 AP >= 0.05 at epoch 25.

---

### Q29. Cross-Dataset Pose Transfer: IKEA ASM Head Pose for Pre-Training

**Context:** The IKEA ASM dataset includes egocentric head pose from a head-mounted camera (similar to HoloLens 2 in IndustReal). The head kinematics during furniture assembly (looking at instructions, looking at the assembly, looking at the tool) follow similar patterns across different assembly tasks. Pre-training the ego-pose head on IKEA ASM head pose data could improve the IndustReal pose MAE from 8.14 deg to 7.0-7.5 deg by providing the backbone with more diverse head-pose training data.

The transfer challenge: IKEA ASM uses a different head-mounted camera setup (not HoloLens 2). The camera-to-head calibration differs. However, the head movement patterns (saccades, fixations, smooth pursuit during assembly) are task-specific rather than hardware-specific. Pre-training on head pose from a different device followed by fine-tuning on HL2 data is analogous to pre-training on synthetic head pose (common in the head pose estimation literature).

**Question:** Would pre-training the ego-pose branch (head FiLM + pose MLP) on IKEA ASM head pose data (12 participants, 24K frames, head orientation labels) for 25 epochs before fine-tuning on IndustReal improve forward MAE from 8.14 to 7.0-7.5 deg and up MAE from 7.06 to 6.0-6.5 deg, because the pose branch learns general assembly-viewing head kinematics (looking patterns) that transfer across assembly tasks?

**Why this matters:** Cross-dataset pose transfer has not been attempted in the IndustReal literature. If successful, it demonstrates that assembly-viewing head kinematics are consistent across products and environments -- a finding with implications for AR-assisted assembly training. Estimated impact: -0.6 to -1.1 deg forward MAE.

**Constraints:** Must obtain IKEA ASM head pose labels. Code changes for pose data integration. 25-epoch pre-training + 25-epoch fine-tuning on 3060 (~5 days). Must handle different coordinate frame conventions between IKEA ASM and HL2. Must freeze backbone during pre-training (to avoid catastrophic forgetting of detection features).

**Hypothesis:** IKEA ASM pre-trained pose branch achieves forward MAE=7.0-7.5 deg at IndustReal fine-tuning epoch 15 (vs 8.14 at epoch 11 for baseline). Up MAE=6.0-6.5 deg. The pre-training benefit is largest for participants with extreme head kinematics (tallest/shortest participants where HL2 mounting angle differs most from median).

**Validation:** Pre-train pose branch on IKEA ASM. Fine-tune on IndustReal. Compare forward/up MAE at fine-tuning epochs 5, 10, 15, 25 against scratch-trained baseline. Success criterion: forward MAE <= 7.5 deg at fine-tuning epoch 15.

---

### Q30. Universal Assembly State Representation via Metric Learning

**Context:** The ultimate goal of cross-dataset transfer is a universal assembly state representation -- backbone features that can distinguish assembly states regardless of the specific product being assembled. Metric learning (contrastive or triplet loss) can learn such representations by pulling same-state features together and pushing different-state features apart, across multiple datasets.

The constraint from binary codes (116-winning-aaiml-synthesis.md:171-186): The IndustReal ASD binary code decomposes assembly state into independent component presence/absence bits. Different assembly products have different components. But the binary decomposition is universal -- any assembly state can be represented as a vector of component bits. Metric learning at the bit level (is component X present or absent?) rather than the state level (is the assembly in state 7 or state 10?) could learn component-presence features that transfer across products.

**Question:** Would metric learning at the component-bit level -- using a contrastive loss that pulls together features of frames where a specific component presence bit matches and pushes apart features where the bit differs, applied jointly to IndustReal (24 components) and IKEA ASM (12 assembly parts) data -- learn a universal component presence feature representation that transfers to novel assembly tasks with zero-shot state detection, and does this representation improve IndustReal mAP50_pc by 0.04-0.08 through better feature separation?

**Why this matters:** A universal assembly state representation would be a significant contribution beyond the current paper scope. It would demonstrate that assembly state is fundamentally a set of independent binary attributes that can be learned across products. Estimated impact: +0.04-0.08 mAP50_pc, plus major academic contribution.

**Constraints:** Code changes for multi-dataset metric learning (~150 lines). 50-epoch joint metric learning + 25-epoch fine-tuning on 5060 Ti (~12 days). Must define component-alignment across datasets (different components in IKEA ASM vs IndustReal -- mapping requires manual correspondence). The metric learning must not degrade per-task performance.

**Hypothesis:** Component-bit metric learning achieves IndustReal mAP50_pc=0.55-0.60 at final epoch. A zero-shot transfer evaluation (training on IKEA ASM + partial IndustReal, testing on new IndustReal components) achieves more than 50% of fully-supervised AP with 10% of the labeled data.

**Validation:** Implement component-bit metric learning. Train on IKEA ASM + IndustReal jointly. Evaluate zero-shot transfer on held-out IndustReal components. Compare mAP50_pc with fully-supervised baseline. Success criterion: zero-shot transfer achieves more than 50% of fully-supervised AP with 10% labeled data.

---


# Category 7: Detection-Specific (5 Questions)

Detection mAP50=0.317 must reach 0.838 (YOLOv8m SOTA). These five questions target specific detection improvements: pretraining strategy, scaling, ablation studies, and fine-tuning approaches that could close this gap.

---

### Q31. Object365 Pretraining for Detection Backbone

**Context:** Current ConvNeXt-Tiny is randomly initialized (117 Q26 analysis). ImageNet pretraining showed -0.02 mAP regression due to catastrophic forgetting (115-execution-plan-to-sota.md:102). Objects365 (Shao et al., 2019, 365 object categories, 2M images, 30M bounding boxes) is the largest detection-specific dataset and provides a much more relevant pretraining domain than ImageNet for assembly state detection. The 365 categories include tools, hardware, and small parts -- directly relevant to IndustReal's components.

The pretraining relevance hierarchy for detection: Objects365 > COCO (80 classes) > ImageNet (1000 classes) > random init. Each step up provides more detection-relevant features. Objects365 features include generic objectness, edge/contour detection, and spatial relationship understanding -- all directly applicable to detecting bolts, washers, nuts, and their assembly states.

**Question:** Would Objects365-pretrained ConvNeXt-Tiny (using timm weights or detectron2 model zoo, with discriminative fine-tuning: frozen backbone at 1e-5 for 5 epochs, then 5e-5 for remaining schedule) improve det_mAP50 from 0.317 to 0.38-0.44 and mAP50_pc from 0.506 to 0.56-0.62, exceeding ImageNet-pretrained performance because Objects365 provides detection-specific features (objectness, edges, parts) that transfer directly to assembly component detection?

**Why this matters:** Objects365 pretraining is standard in detection (used by YOLOv8m, Detectron2 models) but has not been evaluated for assembly state detection. If it provides +0.06-0.12 mAP50 over random init (vs ImageNet's -0.02 regression under poor fine-tuning), it would close 10-20% of the YOLOv8m gap with a single config change. Estimated impact: +0.06-0.12 mAP50.

**Constraints:** Config change (PRETRAINED=True, MODEL_WEIGHTS=objects365_weights.pth). Must verify Objects365 weights are ConvNeXt-Tiny compatible (timm has ConvNeXt-Tiny via detectron2). 50-epoch run on 5060 Ti. Must use proper discriminative LR (backbone 1e-5 for 5 epochs, then 5e-5) to avoid the -0.02 regression observed with ImageNet.

**Hypothesis:** Objects365-pretrained model achieves det_mAP50=0.38-0.44 (vs 0.317) and mAP50_pc=0.56-0.62 (vs 0.506) at epoch 25. Low-GT channels (16, 19, 22) show +0.05-0.12 AP improvement because Objects365 features provide better generalization from sparse examples. The improvement maintains through epoch 50.

**Validation:** Download Objects365-pretrained ConvNeXt-Tiny weights. Run 50 epochs with discriminative LR. Compare epoch-by-epoch detection metrics against random-init baseline and ImageNet-pretrained baseline (if available). Success criterion: det_mAP50 >= 0.38 at epoch 25.

---

### Q32. DINOv2 Self-Supervised Features for Detection

**Context:** DINOv2 (Oquab et al., 2023) produces vision transformer features that have been shown to outperform supervised pretraining for dense prediction tasks including detection and segmentation. The key advantage: DINOv2 features are naturally structured -- they encode object boundaries, part-whole relationships, and spatial correspondence without explicit supervision. For ASD detection where the visual difference between "washer present" and "washer absent" can be a few pixels, DINOv2's structured representations may provide a significant advantage.

The DINOv2 + ConvNeXt adaptation challenge: DINOv2 uses a ViT architecture. Directly using DINOv2 features with our ConvNeXt-Tiny architecture requires either (a) using DINOv2 as a frozen feature extractor with a detection head on top, or (b) distilling DINOv2 features into ConvNeXt-Tiny. Option (a) is simpler: replace ConvNeXt-Tiny with DINOv2-B (86M params, ~307 GMACs) or DINOv2-S (21M params, ~85 GMACs) as the backbone.

**Question:** Would replacing ConvNeXt-Tiny (28.6M) with DINOv2-S (21M, ViT-S, 85 GMACs) as the frozen backbone (no backbone fine-tuning, only task heads trained) improve det_mAP50 from 0.317 to 0.35-0.40 and act_macro_F1 from 0.110 to 0.15-0.20, because DINOv2's self-supervised features provide structured representations of object parts and spatial relationships that are directly applicable to assembly state reasoning -- and does the frozen backbone prevent the catastrophic forgetting observed with ImageNet fine-tuning (the -0.02 mAP regression)?

**Why this matters:** DINOv2 features are a drop-in backbone replacement with a proven track record for dense tasks. The frozen-backbone approach eliminates the fine-tuning instability that plagued ImageNet pretraining. If DINOv2-S (21M params, smaller than ConvNeXt-Tiny's 28.6M) provides detection improvement, it is a strict pareto improvement. Estimated impact: +0.03-0.08 mAP50, +0.04-0.09 act macro-F1.

**Constraints:** Code change to model.py for ViT backbone integration (~80 lines). Must handle different feature pyramid structure (ViT features are at single scale, need to be converted to multi-scale FPN). DINOv2-S fits in 16 GB VRAM at B=4 (21M params, ~85 GMACs). Inference slower than ConvNeXt (~1.5x). Must decide between frozen and fine-tuned backbone (frozen is safer, fine-tuned may be better).

**Hypothesis:** DINOv2-S frozen backbone achieves det_mAP50=0.35-0.40 and act_macro_F1=0.15-0.20 at epoch 25 of head-only training. The ViT features provide better class separation (pred_distinct for activity increases from 35/69 to 45+/69). Frozen backbone eliminates the catastrophic forgetting issue.

**Validation:** Integrate DINOv2-S as backbone with frozen weights. Train only task heads for 50 epochs. Compare all metrics against ConvNeXt-Tiny random init baseline at equivalent epochs. Success criterion: det_mAP50 >= 0.35 at epoch 25.

---

### Q33. Detection-Specific Scaling: Compute-Optimal Model Size for ASD

**Context:** Current ConvNeXt-Tiny (28.6M backbone) is relatively small for a 4-task model (total 46.5M). The compute-optimal scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) suggest that for a fixed compute budget, there is an optimal model size. For our dataset (26K training frames, 4,710 detection GT frames), the optimal model size may be smaller than ConvNeXt-Tiny.

Compute budget analysis: At 0.6 batch/s on 5060 Ti (111-overview.md:241-254), 100 epochs = ~46 hours compute. Given 26K training frames and the parameter-to-sample ratio (1,769:1), the model may be overparameterized for the available supervision. A smaller model with more training epochs per unit time could achieve better performance by completing more training iterations within the same wall-clock budget.

**Question:** Would a lighter backbone (ConvNeXt-Nano, 15.6M params, ~2.2 GMACs, 1.5x faster training) achieve comparable or better det_mAP50 (within 0.02 of ConvNeXt-Tiny) while completing 150 epochs (1.5x more training within same compute budget) -- and does the Nano variant generalize better on low-GT channels (16, 19, 22) because its smaller capacity reduces overfitting to the sparse supervision?

**Why this matters:** The efficiency thesis (116-winning-aaiml-synthesis.md Section 2) positions "single-GPU multi-task" as a core contribution. A Nano backbone that achieves 90% of Tiny's detection performance at 50% fewer parameters and 1.5x faster training would strengthen this narrative. Estimated impact: -0.01 to -0.03 mAP50 (degradation) but +2x speed and stronger efficiency claim.

**Constraints:** Code change (model.py, use ConvNeXt-Nano from timm). 150-epoch run (same wall-clock as 100 epochs Tiny). 15.6M backbone vs 28.6M = 46% fewer backbone parameters. VRAM reduction ~1.5 GB. Must monitor whether 150 epochs of Nano matches 100 epochs of Tiny.

**Hypothesis:** ConvNeXt-Nano achieves det_mAP50=0.30-0.33 (vs 0.317 for Tiny) at 150 epochs. Per-epoch learning is faster (Nano reaches Tiny's epoch-25 mAP at epoch 15-18 due to faster convergence). Activity macro-F1 is comparable (within 0.01). The efficiency thesis is supported by showing Nano achieves 95% of Tiny performance at 46% fewer backbone parameters.

**Validation:** Train ConvNeXt-Nano for 150 epochs. Compare epoch-by-epoch detection metrics against ConvNeXt-Tiny at equivalent compute time (Nano epoch 150 vs Tiny epoch 100). Success criterion: Nano mAP50 within 0.02 of Tiny mAP50 at equivalent compute time.

---

### Q34. Teacher-Student Distillation from YOLOv8m

**Context:** YOLOv8m achieves 0.838 mAP50 (114-comparability-vs-4-papers.md:146-155). Knowledge distillation from YOLOv8m to our ConvNeXt-Tiny detection head could transfer YOLOv8m's detection knowledge without requiring the full YOLOv8m architecture. Specifically, distillation at the feature level (feature imitation) or prediction level (response distillation) would teach our detection head to produce YOLOv8m-like outputs on frames where YOLOv8m is more confident.

**D1 already planned (118 Section 8):** D1 evaluates YOLOv8m on our split (estimated 0.78-0.82 per 117 Q41). Once D1 runs, we have YOLOv8m predictions on all validation frames. These predictions can serve as soft targets for distillation: minimize KL divergence between our detection head logits and YOLOv8m's logits (softened by temperature T=5), weighted by YOLOv8m's confidence. This is standard in detection distillation (e.g., MaskDINO, DINO) and has demonstrated 2-5% AP improvement.

**Question:** Would knowledge distillation from YOLOv8m to our detection head -- minimizing KL divergence between our class logits and YOLOv8m's softened logits (T=5) for the 24 ASD classes, weighted by YOLOv8m's predicted confidence (only confident predictions contribute) -- improve det_mAP50 from 0.317 to 0.35-0.40 by transferring YOLOv8m's class discrimination knowledge to our head, and does this approach outperform pseudo-labeling (Q38) because soft targets preserve uncertainty information that hard pseudo-labels discard?

**Why this matters:** Distillation from a SOTA detector to a multi-task model is a novel approach to closing the detection gap. It leverages YOLOv8m's existing performance (which we can run via D1 in 2 hours) without requiring architectural changes. The soft targets provide more information than pseudo-labels (Q38) because they encode inter-class similarities. Estimated impact: +0.03-0.08 mAP50.

**Constraints:** Code change to losses.py (~30 lines for KL divergence distillation loss). Requires YOLOv8m predictions on training set (run YOLOv8m inference on all 26K training frames). Distillation weight lambda to sweep {0.1, 0.5, 1.0, 2.0}. Temperature T to sweep {3, 5, 10}. 25-epoch ablation on 3060. Must ensure distillation does not interfere with other task gradients (use DETACH_PSR_FPN-style protection for distillation gradients).

**Hypothesis:** YOLOv8m distillation (lambda=0.5, T=5) achieves det_mAP50=0.35-0.40 at epoch 25 (vs 0.317 baseline). The improvement is largest on channels where YOLOv8m is most confident (top-10 channels by YOLOv8m confidence show +0.04-0.08 AP). Low-confidence channels (16, 19) show minimal improvement because YOLOv8m's soft targets are uncertain for these channels too.

**Validation:** Run YOLOv8m inference on training frames. Implement distillation loss. Run 25-epoch ablation with hyperparameter sweep. Compare det_mAP50, per-class AP against non-distillation baseline. Success criterion: det_mAP50 >= 0.35 at epoch 25.

---

### Q35. Multi-Scale Training with Random Crop Augmentation

**Context:** Current training uses fixed 224x224 input size (111-overview.md:95-100). The assembly components vary in scale: bolts appear at ~30px, washers at ~80px, full assembly at ~180px. Multi-scale training (randomly sampling input sizes from {160, 192, 224, 256, 288} at each epoch) is standard in detection (used by YOLOv8m, Detectron2, MMDetection) and provides 2-4% mAP improvement by making the detector scale-invariant.

**The scale variation in ASD (116-winning-aaiml-synthesis.md:171-186):** Channels that differ by small components (e.g., channel 22 vs 10, differing by component 4 which is a small bolt) may be undetectable at 224x224 because the distinguishing component is only a few pixels. Multi-scale training at higher resolution (288x288) would provide more pixels for these small components, while lower resolution (160x160) improves speed and provides scale augmentation.

**Question:** Would multi-scale training -- randomly sampling input size per batch from {160, 192, 224, 256, 288} at each training step, with corresponding anchor box scale adjustment -- improve mAP50_pc from 0.506 to 0.54-0.58 at epoch 50 by making the detector scale-invariant, with the improvement concentrated on small-component channels (22, 16, 19) that benefit from higher-resolution training at 288x288?

**Why this matters:** Multi-scale training is a standard detection technique that is conspicuously absent from our current training recipe. It requires minimal code changes (~10 lines in dataset.py) and has zero inference cost (test at 224x224). The 2-4% mAP improvement documented in the detection literature is entirely additive to other improvements. Estimated impact: +0.03-0.07 mAP50_pc.

**Constraints:** Code change to dataset.py (~10 lines for random scale sampling). 50-epoch run on 5060 Ti. VRAM at 288x288 (B=4) estimated at ~14 GB (within 16 GB). Must adjust anchor boxes proportionally to scale. Cannot be applied mid-training (different anchor scales). Multi-scale training increases epoch time by ~1.2x (larger images slower).

**Hypothesis:** Multi-scale training achieves mAP50_pc=0.54-0.58 at epoch 50 (vs 0.506). Channels 22, 16, 19 show +0.05-0.15 AP improvement because higher-resolution training (288x288) provides more pixels for their distinguishing small components. Top channels (7, 9, 10) maintain AP within 0.01. PSR F1 improves +0.01-0.03 due to better feature quality.

**Validation:** Implement multi-scale training. Run 50 epochs. Compare per-class AP, mAP50_pc against fixed 224x224 baseline. Success criterion: mAP50_pc >= 0.54 at epoch 50.

---

# Category 8: Activity-Specific (5 Questions)

Activity macro-F1=0.110 is 55-69% below the remapped MViTv2 estimate (0.25-0.35 per 117 Q45). These five questions explore temporal modeling, verb-grouping, sampling strategies, and architectural changes specific to activity recognition.

---

### Q36. Temporal Head with Dilated Convolutions and Hierarchical Pooling

**Context:** Current activity head is a per-frame MLP (0.7M params, 112-training-metrics-deep-dive.md:198-210). The temporal head (T2 in 115-execution-plan-to-sota.md:520-558) uses 2-layer TCN on 16-frame windows. MViTv2 uses 16-frame temporal clips with multi-scale transformer pooling. A dilated TCN with hierarchical temporal pooling (similar to TSN/TSM) could provide activity recognition improvement without full transformer complexity.

The design principle from Q7 (117): The TCN receptive field must cover the average action duration (19 frames at 10 fps). A 4-layer TCN with dilations {1, 2, 4, 8} provides 31-frame receptive field. Hierarchical pooling (average pool every 4 frames, then apply another temp conv) reduces temporal resolution while building higher-level temporal abstractions.

**Question:** Would a hierarchical temporal head combining 4-layer dilated TCN (dilations 1, 2, 4, 8, kernel=3, channels=256) with 2-level temporal pooling (stride-4 after TCN layer 2, then stride-2 after TCN layer 4) improve temporal macro-F1 from an estimated 0.15 (T2 baseline, 2-layer TCN) to 0.22-0.28 by providing sufficient receptive field (31 frames) and hierarchical temporal abstraction -- and does the hierarchical architecture outperform the flat TCN because different actions have different temporal extents (0.5s for "grasp" vs 3s for "tighten") that benefit from multi-scale temporal features?

**Why this matters:** The T2 temporal head (current planned temporal architecture) uses a flat 2-layer TCN with 7-frame RF -- barely covering one third of the mean action duration. A properly designed temporal architecture (dilated + hierarchical) could more than double the temporal macro-F1 estimate, making the T2 investment worthwhile. Estimated impact: +0.07-0.13 temporal macro-F1 over 2-layer TCN.

**Constraints:** Code change to activity head (~80 lines for dilated TCN + hierarchical pooling). Added parameters ~2M (TCN + pooling). 50-epoch temporal training on 5060 Ti (~7 days). Must coordinate with T2 go/no-go gate (118 Section 5, G1). If remapped MViTv2 lands at 0.25 (lower end of Q45 estimate), this architecture would achieve 88-112% of SOTA.

**Hypothesis:** Hierarchical dilated TCN achieves temporal macro-F1=0.22-0.28 at epoch 50. Short actions (grasp, pick up: 0.5s) benefit from fine-scale features (TCN layer 1-2). Long actions (tighten, screw: 3s) benefit from coarse-scale features (TCN layer 3-4 + pooling). Per-class F1 shows more uniform distribution across action durations vs flat TCN.

**Validation:** Implement hierarchical dilated TCN. Run 50 epochs. Compare temporal macro-F1 against 2-layer TCN baseline at equivalent epochs. Success criterion: temporal macro-F1 >= 0.22 at epoch 50.

---

### Q37. Verb-Grouping with Learned Class Hierarchy

**Context:** Activity head predicts 69 verb-grouped classes (reduced from 75 fine-grained) per 111-overview.md:68-69. The verb-grouping strategy merges verbs by their action type: e.g., "tighten_screw", "tighten_bolt", "tighten_nut" all become "tighten." This reduces class count but may lose discriminative information. A learned class hierarchy (like HD-CNN or label embedding trees) could exploit the verb-noun structure without discarding the noun information.

**The improved approach:** Instead of hard verb-grouping (75 to 69), use a soft hierarchy: predict both verb (7 verb types: tighten, loosen, insert, remove, check, grasp, idle) and noun (20+ object types: screw, bolt, washer, nut, tool_x, etc.) jointly, then combine into a final 69-class prediction through learned combination weights. This preserves the fine-grained discriminative signal of nouns while still reporting verb-grouped metrics for comparability.

**Context from per-class AP analysis (112-training-metrics-deep-dive.md:497-514):** Activity pred_distinct = 35/69 classes (only 35 classes are ever predicted). The 34 unpredicted classes are likely fine-grained noun distinctions that the model cannot learn because it never sees enough examples of "tighten_screw" vs "tighten_bolt" to distinguish them. A soft hierarchy where both verb and noun contribute to the prediction could help by sharing noun information across verbs.

**Question:** Would a two-level hierarchical activity head -- predicting verb class (7 classes) and noun class (20+ classes) separately using two MLP branches on shared features, then combining via learned bilinear interaction weights to produce the final 69-class prediction -- improve macro-F1 from 0.110 to 0.15-0.20 by exploiting the verb-noun compositional structure, and does this approach outperform the current flat 69-class classifier because the 7-verb classification is easier (more training examples per verb) and the 20-noun classification provides auxiliary supervision?

**Why this matters:** The verb-grouping strategy is currently a hard merge. A soft hierarchical approach is more principled -- it preserves all information while exploiting the compositional structure. This is a methodological contribution (hierarchical activity recognition for assembly tasks) that no prior IndustReal paper explores. Estimated impact: +0.04-0.09 macro-F1.

**Constraints:** Code change to activity head (~60 lines for hierarchical branches). Adds ~0.5M params. 50-epoch run on 5060 Ti. Must define verb-to-noun mapping from dataset taxonomy. Must handle verbs without nouns (e.g., "idle" has no object). The bilinear combination weight can be a learned 7x20 matrix (140 params).

**Hypothesis:** Hierarchical activity head achieves macro-F1=0.15-0.20 at epoch 50. Verb classification accuracy reaches 70-80% (vs 35/69 predicted = ~50% effective). The learned bilinear weights show interpretable verb-noun combinations (e.g., "tighten" strongly weights screw/bolt nouns). pred_distinct increases from 35/69 to 50+/69.

**Validation:** Implement hierarchical verb-noun head. Run 50 epochs. Compare macro-F1, verb-only accuracy, pred_distinct against flat 69-class baseline. Success criterion: macro-F1 >= 0.15 at epoch 50.

---

### Q38. Pseudo-Labeling for Activity: Leveraging Detection Features

**Context:** Activity head currently uses backbone features shared with detection. Detection is more reliable (mAP50=0.317, yes low, but detection predictions are discriminative for certain activity classes -- e.g., "component 5 is visible" strongly predicts "tighten component 5"). Detection predictions can serve as auxiliary features for activity recognition, similar to how object detections are used as input to activity recognition models in standard action recognition pipelines.

The detection-activity correlation (from assembly logic): When the detection head predicts channel 7 (component X present), the activity class is almost certainly "tighten X", "check X", or "loosen X." The detection output provides a strong prior on which components are currently being manipulated. Currently, this information is only available through the shared backbone (gradient-mediated) and not explicitly provided to the activity head.

**Question:** Would augmenting the activity head input with detection prediction logits (concatenating the 24-dimensional detection logits to the backbone features before the activity MLP) improve activity macro-F1 from 0.110 to 0.14-0.19 by providing explicit component-presence information that reduces the activity head's ambiguity about which object is being manipulated -- and does this cross-task feature sharing outperform the current implicit sharing through backbone gradients alone?

**Why this matters:** Cross-task feature sharing (detection -> activity) is a specific example of multi-task synergy that the paper could highlight. If detection predictions improve activity recognition, it demonstrates tangible multi-task benefit beyond gradient sharing. Estimated impact: +0.03-0.08 macro-F1.

**Constraints:** Code change to model.py (~15 lines for detection logit concatenation). 25-epoch ablation on 3060. Must use stop-gradient on detection logits (prevent activity gradients from affecting detection). VRAM increase negligible (24-dimensional concatenation). Must evaluate whether detection noise hurts activity (if detection logits are wrong, they may confuse the activity head).

**Hypothesis:** Detection-augmented activity head achieves macro-F1=0.14-0.19 at epoch 25 (vs 0.110). Tool-manipulation actions (tighten, loosen) show largest improvement (+0.05-0.10) because detection provides the "which component is being manipulated" signal. Static actions (check, idle) show minimal improvement because they lack object interaction.

**Validation:** Implement detection-feature augmented activity head. Run 25-epoch ablation. Compare macro-F1, per-class F1 against baseline activity head. Success criterion: macro-F1 >= 0.14 at epoch 25.

---

### Q39. Long-Tail Activity Sampling with Synthetic Minority Oversampling

**Context:** Activity classes follow a long-tail distribution: top-10 classes (tighten-related) account for ~70% of frames, bottom-30 classes (idle, check variants, rare actions) account for <5% of frames (112-training-metrics-deep-dive.md:311-337). Standard oversampling (repeating rare class examples) leads to overfitting. SMOTE (Synthetic Minority Oversampling, Chawla et al., 2002) generates synthetic examples by interpolating between rare-class feature vectors.

**SMOTE for activity features:** For two feature vectors f_i and f_j from the same rare activity class, SMOTE generates a new example: f_new = f_i + lambda * (f_j - f_i), where lambda is sampled from U[0, 1]. The synthetic feature is fed to the activity MLP with the original class label. This provides the activity head with more varied examples of rare classes without duplicating real examples.

**Context from gradient analysis (112-training-metrics-deep-dive.md:1288-1300, Appendix B):** Rare activity classes contribute minimal gradient because they appear in few batches. SMOTE would increase their representation by generating synthetic feature vectors, providing more gradient updates for the rare-class MLP weights.

**Question:** Would SMOTE synthetic oversampling for rare activity classes (bottom-30 classes, SMOTE applied at the feature level between two backbone feature vectors from the same rare class, generating 5x synthetic examples per rare class per epoch) improve activity macro-F1 from 0.110 to 0.13-0.17 by providing the activity MLP with more training examples for rare classes, and does this approach outperform simple frame-level oversampling (repeating rare class frames) because SMOTE generates diverse synthetic examples that prevent overfitting?

**Why this matters:** Long-tail activity recognition is a known challenge in egocentric vision. SMOTE has been applied to activity recognition but not to IndustReal-specific assembly actions. If it works, it is a simple config-level change with potentially significant impact. Estimated impact: +0.02-0.06 macro-F1.

**Constraints:** Code change to train.py (~30 lines for SMOTE feature generation). 25-epoch ablation on 3060. Must apply SMOTE to backbone features (post-backbone, pre-activity-MLP). Must monitor that SMOTE does not generate implausible feature combinations (interpolation between two different rare classes could produce confused features). SMOTE ratio to sweep {2x, 5x, 10x} per rare class.

**Hypothesis:** SMOTE (5x ratio for bottom-30 classes) achieves macro-F1=0.13-0.17 at epoch 25. The bottom-30 classes individually show +0.03-0.10 F1 improvement. pred_distinct increases from 35/69 to 40+/69. Top-10 classes maintain F1 within 0.02 of baseline.

**Validation:** Implement SMOTE for rare activity classes. Run 25-epoch ablation with ratio sweep. Compare macro-F1, per-class F1 against no-oversampling baseline. Success criterion: macro-F1 >= 0.13 at epoch 25.

---

### Q40. VideoMAE Pretraining for Activity Backbone

**Context:** VideoMAE (Tong et al., NeurIPS 2022) extends MAE to video by masking out spacetime cubes and reconstructing the masked patches. For activity recognition, VideoMAE pretraining on Kinetics-400 or Something-Something-v2 provides spatiotemporal features that transfer well to downstream action recognition tasks. Our backbone currently receives no temporal pretraining -- it is trained from scratch on static frames.

The VideoMAE adaptation: Since our backbone is ConvNeXt-Tiny (a CNN, not ViT), standard VideoMAE (which uses ViT) must be adapted. Options: (a) use a ViT-T backbone with VideoMAE pretraining, or (b) distill VideoMAE features into ConvNeXt-Tiny. Option (a) is simpler: replace ConvNeXt-Tiny with VideoMAE-pretrained ViT-T (5.7M params, comparable to ConvNeXt-Tiny's 28.6M) for the activity branch only.

**Question:** Would a VideoMAE-pretrained ViT-T backbone for the activity branch only (keeping ConvNeXt-Tiny for detection/pose/PSR) -- processing 16-frame clips with VideoMAE spatiotemporal features, combined with per-frame ConvNeXt features through late fusion -- improve temporal macro-F1 from an estimated 0.15 (T2 baseline) to 0.25-0.35, matching the remapped MViTv2 estimate (117 Q45), because the VideoMAE pretrained spatiotemporal features provide action-specific temporal structure that our scratch-trained ConvNeXt cannot learn?

**Why this matters:** VideoMAE is the current SOTA video pretraining method for action recognition. A dual-backbone architecture (ConvNeXt-Tiny for static tasks + ViT-T VideoMAE for activity) would be novel for multi-task assembly understanding. If it matches or exceeds MViTv2 performance, it would be a headline contribution. Estimated impact: +0.10-0.20 temporal macro-F1 over scratch-trained temporal head.

**Constraints:** Code change to model.py for dual-backbone architecture (~100 lines). ViT-T VideoMAE has 5.7M params (smaller than ConvNeXt-Tiny's 28.6M). 16-frame clip processing increases VRAM by ~3 GB (ViT-T activations). Must download VideoMAE pretrained weights. 50-epoch temporal training on 5060 Ti (~8 days). The dual-backbone design doubles inference compute for activity.

**Hypothesis:** VideoMAE-pretrained dual-backbone achieves temporal macro-F1=0.25-0.35 at epoch 50, matching MViTv2 remapped performance. The static ConvNeXt branch provides per-frame detection/pose quality unchanged (within 0.01 of single-backbone). Detection/PSR metrics are unaffected because they use only the ConvNeXt branch.

**Validation:** Implement dual-backbone architecture. Run 50-epoch temporal training. Compare temporal macro-F1 against single-backbone temporal head and MViTv2 remapped estimate (from Q45/T3). Success criterion: temporal macro-F1 >= 0.25 at epoch 50.

---


# Category 9: Ego-Pose-Specific (5 Questions)

Forward MAE=8.14 deg, up MAE=7.06 deg (111-overview.md:751-757). The estimated HoloLens 2 sensor noise floor is 5-7 deg (116-winning-aaiml-synthesis.md:573). These five questions explore how to push ego-pose below 7 deg and toward the noise floor, using geodesic loss, rotation augmentation, multi-scale pose heads, temporal smoothing, and uncertainty modeling.

---

### Q41. Geodesic Loss with Rotation Representation Comparison (6D vs Quaternion vs 3D Vector)

**Context:** Current pose head predicts 3D unit vectors (forward, up) with MSE loss (112-training-metrics-deep-dive.md:721-749). Q11 proposes geodesic loss (direct angle between predicted and ground-truth vectors). However, the rotation representation itself matters: 6D continuous rotation representation (Zhou et al., CVPR 2019) predicts a full 3x3 rotation matrix (first two rows) and is theoretically better than predicting unit vectors because it enforces orthonormality. Q11's geodesic loss on unit vectors may still suffer from the representation gap -- unit vectors are not a complete rotation representation (they specify only 2 of 3 rotation axes).

**The rotation representation hierarchy for head pose:** (1) 3D unit vectors (current) -- simple but incomplete rotation representation. (2) Quaternions (4D, unit-norm constrained) -- complete representation but discontinuous at antipodal points. (3) 6D rotation (Zhou et al.) -- continuous, complete representation with no singularities, recommended for head pose estimation. The 6D representation plus geodesic loss on the reconstructed rotation matrix is the current SOTA approach for head pose estimation (e.g., 6DRepNet).

**Question:** Would a 6D continuous rotation representation (predicting a 3x2 matrix, completing to 3x3 via Gram-Schmidt, formulating the geodesic loss as the angular error between predicted and GT rotation matrices) improve forward MAE from 8.14 to 6.5-7.0 deg and up MAE from 7.06 to 5.5-6.0 deg, outperforming both the current 3D vector+MSE and the proposed 3D vector+geodesic (Q11) because the 6D representation provides a complete rotation parameterization without discontinuities?

**Why this matters:** The 6D rotation representation is a standard improvement in head pose estimation (6DRepNet achieves 3.6 deg MAE on 300W-LP, vs 4.8 deg for quaternion-based methods). If it reduces our MAE by 1.0-1.6 deg, the combined improvements (6D + uncertainty weighting Q13 + position removal Q12) could push ego-pose below 6.5 deg. Estimated impact: -0.8 to -1.6 deg forward MAE.

**Constraints:** Code change to model.py (rotation head output 6D) + losses.py (geodesic loss on rotation matrices). 25-epoch ablation on 3060. Must ensure the 6D-to-3x3 Gram-Schmidt is numerically stable (standard implementation exists in PyTorch3D or can be implemented in 10 lines). Additional 3 output dimensions (6 vs current 2x3=6, same output dimension).

**Hypothesis:** 6D rotation representation + geodesic loss achieves forward MAE=6.5-7.0 deg and up MAE=5.5-6.0 deg at epoch 25. The improvement over 3D vector+geodesic (estimated 7.0-7.5 deg from Q11) is 0.3-0.8 deg, attributable to the complete rotation representation.

**Validation:** Implement 6D rotation head. Run 25-epoch ablation. Compare forward/up MAE against 3D vector+MSE (current) and 3D vector+geodesic (Q11). Success criterion: forward MAE <= 7.0 deg at epoch 25.

---

### Q42. Temporal Head Pose Filtering with Kalman Smoothing

**Context:** Ego-pose predictions are per-frame independent (111-overview.md:630-640). Head movement is smooth and continuous -- a Kalman filter could smooth predictions by modeling head kinematics (position, velocity, acceleration) and reducing per-frame noise. The current 8.14 deg MAE includes measurement noise from both the sensor and the model; temporal filtering addresses the latter.

The Kalman filter advantage: Head pose follows a constant-velocity or constant-acceleration model within short temporal windows. Per-frame prediction noise (std ~3-5 deg empirically from the model) can be reduced by the Kalman smoothing factor R = measurement_noise / process_noise. A simple 1D Kalman filter per rotation axis (3 independent Kalman filters for yaw, pitch, roll) or a full 3D rotation Kalman filter on SO(3) manifold could reduce MAE by 0.3-0.8 deg.

**Context from temporal activity (118 Section 1, 7.2-second):** The current system has no temporal processing for any head. A Kalman filter for pose requires no training, no architecture changes, and can be applied as post-processing on existing predictions. This is analogous to temporal smoothing for PSR (Q19) but for pose.

**Question:** Would a Kalman smoother applied to per-frame head pose predictions (2 independent Kalman filters on forward and up unit vectors, process noise Q=1e-4, measurement noise R=1e-2, estimated via offline EM on training set) reduce forward MAE from 8.14 to 7.5-7.8 deg and up MAE from 7.06 to 6.5-6.8 deg by reducing per-frame prediction noise -- and does the Kalman filter interact positively with geodesic loss (Q41, expected forward 6.5-7.0) by further reducing the noise floor to 6.2-6.7 deg combined?

**Why this matters:** Temporal smoothing for ego-pose is a zero-cost post-processing improvement. The Kalman filter adds no inference overhead at test time when implemented as online filtering. It could push the combined improvements (6D + Kalman + uncertainty) toward the HL2 noise floor of 5-7 deg. Estimated impact: -0.3 to -0.8 deg forward MAE.

**Constraints:** Code change to evaluate.py (~30 lines for Kalman filter). No training required. Must tune process noise Q and measurement noise R on validation set. Can be applied as post-processing on any checkpoint. Must handle SO(3) boundary conditions (rotation wrap-around at +/-180 deg).

**Hypothesis:** Kalman smoothed predictions achieve forward MAE=7.5-7.8 deg (vs 8.14 raw) and up MAE=6.5-6.8 deg (vs 7.06 raw) at epoch 11. Combined with 6D rotation representation (estimated 6.5-7.0), the Kalman filter further improves to 6.2-6.7 deg. The filter is most effective on participants with high per-frame variance (unstable head movements).

**Validation:** Implement Kalman filter as post-processing. Apply to epoch-11 checkpoint predictions. Compare smoothed vs raw MAE. Success criterion: Kalman reduces forward MAE by >= 0.3 deg.

---

### Q43. Multi-Scale Pose Head with Coarse-to-Fine Refinement

**Context:** Current pose head uses a single MLP processing final backbone features (112-training-metrics-deep-dive.md:198-210). A multi-scale pose head that processes features from multiple backbone stages (P2, P3, P4) could improve accuracy by combining fine spatial details (P2: edge orientations, head boundaries) with coarse semantic features (P4: head context, body position).

The coarse-to-fine principle in head pose estimation: Coarse scale (P4, 14x14) provides head location and rough orientation. Fine scale (P2, 56x56) provides precise edges and local features for final refinement. A two-stage head: first predict coarse pose from P4 features (MLP, 128 hidden), then refine using P2 features (MLP, 256 hidden with coarse pose as additional input). This adds ~0.3M params but provides explicit multi-scale processing.

**Context from gradient analysis (112-training-metrics-deep-dive.md:845-870, HP_PREC_CAP):** The pose head is HP_PREC_CAPped at 1.25x precision. Increasing effective gradient through architectural improvement (better feature utilization) is a complementary approach to removing the cap (Q24).

**Question:** Would a coarse-to-fine multi-scale pose head -- first predicting coarse pose from P4 features (1 MLP layer, 128D), then refining with P2 features (2 MLP layers, 256D, concatenating coarse pose as additional input) -- improve forward MAE from 8.14 to 7.3-7.8 deg by leveraging both fine-scale edge information and coarse-scale context, and does this architectural improvement provide additive gains to geodesic loss (Q41) and Kalman filtering (Q42)?

**Why this matters:** Multi-scale pose regression is standard in face alignment (HRNet, etc.) but not applied to egocentric head pose estimation. This architectural change is complementary to loss function and post-processing improvements, providing a potential cumulative gain of 1.5-2.5 deg when combined. Estimated impact: -0.3 to -0.8 deg forward MAE.

**Constraints:** Code change to model.py (~40 lines for multi-scale head). Adds ~0.3M params. 25-epoch ablation on 3060. VRAM increase negligible. Must ensure P2 features are not too noisy (fine scale may contain irrelevant detail). Coarse pose dimension to sweep {3, 6, 9}.

**Hypothesis:** Multi-scale coarse-to-fine head achieves forward MAE=7.3-7.8 deg at epoch 25 (vs 8.14 baseline). The improvement is additive to geodesic loss (6D + geodesic estimated 6.5-7.0 + multi-scale estimated 6.3-6.8 combined). High-frequency head movements benefit most from P2 fine-scale refinement.

**Validation:** Implement multi-scale pose head. Run 25-epoch ablation. Compare forward/up MAE against single-scale baseline. Success criterion: forward MAE <= 7.8 deg at epoch 25.

---

### Q44. Ego-Pose Data Augmentation: Synthetic Head Movement from IKEA ASM

**Context:** The 12 IndustReal participants provide limited head movement diversity (similar assembly-watching patterns). IKEA ASM provides 12 additional participants with different furniture, viewpoints, and head kinematics. Q29 proposes pre-training on IKEA ASM head pose. An alternative, more targeted approach: use IKEA ASM head pose sequences to augment IndustReal images by overlaying synthetic head movements.

The augmentation mechanism: For each IndustReal image with a known head pose (forward, up vectors), find the nearest IKEA ASM head pose sequence (by forward/up angle) and replace the current pose with a nearby pose from the IKEA ASM trajectory. This introduces realistic head movement patterns from furniture assembly (looking down to check, looking up for instruction, looking sideways for tool) while maintaining the IndustReal visual appearance.

**Question:** Would head pose augmentation using IKEA ASM pose trajectories -- for each IndustReal training frame, randomly replacing the current pose with a nearby pose from a matched IKEA ASM participant trajectory, thereby increasing effective head pose diversity by 2-3x -- improve forward MAE from 8.14 to 7.0-7.5 deg by training the model on more diverse head kinematics, and does this augmentation outperform simple random rotation augmentation (Q14, +-15 deg) because it uses realistic assembly-viewing head movement patterns rather than random noise?

**Why this matters:** Cross-dataset pose augmentation is novel -- using one dataset's pose distribution to augment another dataset's training. The diversity of IKEA ASM head kinematics (furniture assembly vs bolt assembly has different viewing patterns) provides complementary training signal. Estimated impact: -0.6 to -1.1 deg forward MAE.

**Constraints:** Code change to dataset.py (~50 lines for pose augmentation). Must pre-compute IKEA ASM pose trajectory database. Offline computation ~1 day. Online augmentation during training adds ~10% data loading overhead. Must ensure pose augmentation is geometrically consistent (image and label should match -- but we are only changing the label, so consistency is automatic for a given image-pose pair).

**Hypothesis:** IKEA ASM pose augmentation achieves forward MAE=7.0-7.5 deg at epoch 25. Participants with extreme head kinematics (tall/short, where our training data is sparse) show the largest improvement (0.5-1.0 deg). Standard-height participants show 0.2-0.5 deg improvement. The augmentation is more effective than random rotation (Q14) because it uses realistic head movement patterns.

**Validation:** Implement IKEA ASM pose augmentation. Run 25-epoch ablation. Compare forward/up MAE against no-augmentation baseline and random rotation baseline (Q14). Success criterion: forward MAE <= 7.5 deg at epoch 25.

---

### Q45. Uncertainty Calibration for Ego-Pose Predictions

**Context:** Ego-pose predictions currently output point estimates (forward, up vectors) with no uncertainty quantification (111-overview.md:751-757). For practical applications (AR assembly guidance), knowing when the pose prediction is reliable is as important as the prediction itself. Deep ensemble uncertainty (Lakshminarayanan et al., 2017) trains K models with different random seeds and uses their variance as prediction uncertainty. MC Dropout (Gal & Ghahramani, 2016) uses dropout at inference to estimate uncertainty.

**The practical value:** In an AR assembly guidance system, the model should indicate when it is uncertain about head pose (e.g., during rapid head movements, or when the visual features are ambiguous). A calibrated uncertainty estimate enables the AR system to fall back to IMU-based pose estimation when vision-based pose is uncertain.

**Question:** Would MC Dropout uncertainty estimation (dropout rate=0.1 in pose MLP, 20 stochastic forward passes at inference, compute mean as prediction and standard deviation as uncertainty) improve forward MAE from 8.14 to 7.5-8.0 deg through model averaging (20 passes average reduces variance by factor of sqrt(20) ~ 4.5x) while also providing calibrated uncertainty estimates where the predicted uncertainty correlates with actual error (correlation coefficient > 0.5) -- and can this uncertainty be used to detect out-of-distribution head poses?

**Why this matters:** Uncertainty calibration for egocentric head pose is unexplored. Providing calibrated uncertainty alongside point predictions would be a methodological contribution for AR applications. The MAE improvement from MC averaging is a side benefit. Estimated impact: -0.1 to -0.6 deg forward MAE (from averaging), +uncertainty calibration contribution.

**Constraints:** Code change to model.py (add dropout to pose MLP) and evaluate.py (MC Dropout inference loop). 25-epoch ablation on 3060. Inference cost increases 20x (20 passes). Must tune dropout rate {0.05, 0.1, 0.2}. Must evaluate uncertainty calibration using reliability diagrams and expected calibration error (ECE).

**Hypothesis:** MC Dropout (rate=0.1, 20 passes) achieves forward MAE=7.5-8.0 deg (averaging reduces 0.1-0.6 deg) with ECE < 0.1 (well-calibrated). Predicted uncertainty strongly correlates with actual error during rapid head movements (correlation > 0.5). Uncertainty is higher for participants with extreme head kinematics, detecting OOD poses.

**Validation:** Implement MC Dropout. Run 25-epoch ablation. Compare forward MAE (MC average) vs point estimate baseline. Evaluate uncertainty calibration (ECE, reliability diagram). Success criterion: MC average achieves forward MAE <= 8.0 deg AND ECE < 0.15.

---

# Category 10: PSR-Specific (5 Questions)

PSR POS=0.968 beats SOTA (0.812) by 19%. PSR F1=0.144 trails SOTA (0.901) by 84%. These five questions explore paradigm comparisons, thresholding strategies, temporal models, and evaluation refinements that could improve F1 toward SOTA while maintaining the POS advantage.

---

### Q46. Transition Detection Paradigm: MonotonicDecoder vs Online Detection

**Context:** Current PSR uses the fill-forward (MonotonicDecoder) paradigm: per-frame component predictions are monotonically increasing (once a component is detected as present, it stays present). This achieves high POS (0.968) because the ordering is enforced. But it suffers on F1 (0.144) because delayed detections cascade. An alternative paradigm: detect component transitions (state changes) directly using a temporal model, similar to B3's transformer encoder and STORM-PSR's multi-modal temporal stream.

The transition detection approach: Instead of predicting per-frame component presence and applying fill-forward, predict "component h_i transitioned from absent to present at frame t" directly using a temporal model (TCN or transformer on backbone features). This eliminates the fill-forward delay cascade -- the transition model learns to detect the exact frame of component installation.

**Context from D4 (118 Section 8):** D4 feeds YOLOv8m ASD outputs through MonotonicDecoder (estimated F1 0.45-0.65). If D4 F1 is closer to 0.45 than 0.65, it means even perfect detection inputs are limited by the fill-forward paradigm, and a paradigm change to transition detection is necessary.

**Question:** Would replacing the MonotonicDecoder (fill-forward) with a transition detection temporal head (4-layer TCN on backbone features followed by per-component sigmoid transition classifier, trained to predict transition frames rather than per-component presence) improve PSR F1 from 0.144 to 0.35-0.55 by eliminating the fill-forward delay cascade, at the cost of POS dropping from 0.968 to 0.88-0.92 (still above SOTA 0.812) -- and does the transition detection paradigm close the gap to B3 (0.883 F1) and STORM-PSR (0.901 F1) more effectively than improving detection quality alone (D4)?

**Why this matters:** The MonotonicDecoder vs transition detection paradigm question determines the entire PSR strategy. If transition detection achieves F1 > 0.50 while maintaining POS > 0.88 (beating SOTA on both), it is a clear win. If transition detection fails (F1 < 0.20), the fill-forward paradigm is confirmed as the correct choice. This is the most important PSR design decision. Estimated impact: +0.20-0.40 PSR F1 (if transition detection succeeds).

**Constraints:** Code change to psr_transition.py (~100 lines for transition detection head). 50-epoch run on 5060 Ti. New head adds ~1M params. Must compute transition labels from existing component presence labels (frame-level diff: t_i = 1 if component i present at frame t but not at frame t-1). Must handle multi-component transitions at same frame.

**Hypothesis:** Transition detection head achieves PSR F1=0.35-0.55 and POS=0.88-0.92 at epoch 50. The F1 improvement is concentrated on rapid assembly sequences where consecutive transitions occur within 3 frames (fill-forward's cascading delay is largest). POS remains > 0.88 because the transition model learns component ordering even without explicit fill-forward constraint.

**Validation:** Implement transition detection head. Run 50 epochs. Compare PSR F1, POS, edit against MonotonicDecoder baseline. Compare against D4 (YOLOv8m + MonotonicDecoder) result to determine if paradigm change or detection improvement is more effective. Success criterion: PSR F1 >= 0.35 with POS >= 0.88.

---

### Q47. Temporal PSR with Cross-Attention on Detection Features

**Context:** PSR decoder currently processes each frame's detection features independently (112-training-metrics-deep-dive.md:323-337). A cross-attention temporal model (transformer encoder on sequence of detection features) could learn temporal dependencies between frames -- specifically, how the current assembly state constrains the next possible state.

The assembly state grammar: Given state s(t) with components present {h0, h1, h2}, the next possible states are: (a) add any absent component that is next in the canonical order, or (b) remove any present component if the assembly state allows disassembly. This state transition grammar is a hidden Markov model. A transformer with learned positional encodings can implicitly learn this grammar from data.

**Context from tau analysis (118 Q17, Section 8):** The detection delay tau for rare components (h4, h7-h10) is 3-5+ frames. A temporal model that processes a window of past detection features could detect component presence earlier by recognizing the precursor patterns (e.g., the hand approaching the screw before touching it, which the per-frame model misses because the screw is not yet being manipulated).

**Question:** Would a temporal PSR model with cross-attention (single-layer transformer encoder on a 32-frame window of detection features, with learned positional encoding, followed by per-frame component prediction with fill-forward constraint) improve PSR F1 from 0.144 to 0.25-0.40 by leveraging temporal context to detect components earlier (reducing tau by 1-3 frames for rare components h4, h7-h10) -- and does the cross-attention mechanism outperform a simple TCN because it can attend to the specific temporal positions (pre-grasp, grasp, manipulate, release) that are most informative for each component?

**Why this matters:** Temporal models for PSR have been explored (B3 uses transformer, STORM uses LSTM) but not with cross-attention on detection features. The 32-frame window captures the full assembly context. If successful, this demonstrates learnable temporal reasoning for assembly state that outperforms both fill-forward and simple TCN approaches. Estimated impact: +0.10-0.25 PSR F1.

**Constraints:** Code change to psr_transition.py (~80 lines for transformer encoder). 50-epoch run on 5060 Ti. Adds ~2M params (single-layer transformer with 4 heads, 256 dim). VRAM increase ~1 GB for activation storage (32-frame window). Must process sequences of 32 frames, adding 32-frame batch dimension.

**Hypothesis:** Temporal cross-attention PSR achieves PSR F1=0.25-0.40 at epoch 50. Rare components h4, h7-h10 show 2-4 frame tau reduction (from 3-5 frames to 1-2 frames). POS remains above 0.95 because the fill-forward constraint is still applied after temporal processing. The cross-attention weights show interpretable attention to pre-grasp frames for each component.

**Validation:** Implement temporal cross-attention PSR. Run 50 epochs. Compare PSR F1, per-component F1, tau against per-frame (no temporal) baseline. Success criterion: PSR F1 >= 0.25 at epoch 50.

---

### Q48. Per-Component Adaptive Threshold with Temporal Consistency

**Context:** Q18 proposes per-component confidence thresholds to improve PSR F1 (estimated from 0.144 to 0.17-0.22). However, static thresholds do not account for temporal consistency -- once a component is detected, it should remain present (this is what fill-forward does). A threshold with temporal hysteresis: use a higher threshold for initial detection (to reduce false positives) and a lower threshold for continued presence (to reduce false negatives from jittery predictions).

The hysteresis mechanism: For each component, maintain state (present/absent). When absent: require confidence > tau_high to transition to present. When present: require confidence < tau_low to transition to absent. This is standard in signal processing (Schmitt trigger) and directly addresses the PSR F1 failure mode: noisy predictions cause components to flicker in/out of the present state.

**Context from temporal smoothing (117 Q19, PSR_TEMPORAL_SMOOTH_WEIGHT=0.05):** Current smoothing weight of 0.05 is weak. Hysteresis thresholding provides stronger temporal consistency without the over-smoothing risk of increasing the weight.

**Question:** Would per-component thresholding with temporal hysteresis (tau_high = 0.6 for all components, tau_low = 0.4 for all components, sweep per-component variants) improve PSR F1 from 0.144 to 0.20-0.28 by reducing false negatives from jittery per-frame predictions while maintaining low false positives -- and does the hysteresis mechanism outperform both single-threshold (Q18) and temporal smoothing (Q19) by providing strong temporal consistency without requiring any training or architecture changes?

**Why this matters:** Hysteresis thresholding is a zero-cost inference-only improvement (no training, no architecture change). It directly addresses the noisy-prediction failure mode that dominates PSR errors. The combination of per-component thresholds (Q18) + hysteresis (Q48) could provide +0.05-0.12 F1 from inference-only changes. Estimated impact: +0.06-0.14 PSR F1.

**Constraints:** Code change to psr_transition.py (~20 lines for hysteresis). Inference only on existing predictions. Must sweep tau_high {0.4, 0.5, 0.6, 0.7} and tau_low {0.1, 0.2, 0.3, 0.4} on held-out validation data. Must ensure hysteresis does not lock the decoder in incorrect states (if a false positive occurs, the high threshold makes it harder to recover).

**Hypothesis:** Hysteresis thresholding (tau_high=0.6, tau_low=0.3, uniform across components) achieves PSR F1=0.20-0.28. The improvement is concentrated on components with jittery predictions (likely h7-h10 with low gradient RMS). POS drops slightly (from 0.968 to 0.94-0.96) due to slight detection delays from hysteresis, but remains above SOTA (0.812).

**Validation:** Implement hysteresis thresholding. Grid-search thresholds on validation set. Compare PSR F1, POS against single-threshold baseline (Q18) and temporal smoothing (Q19). Success criterion: PSR F1 >= 0.20 with POS >= 0.94.

---

### Q49. Two-Stage PSR: Detection Quality Predictor + Adaptive Decoder

**Context:** Current MonotonicDecoder applies the same processing regardless of detection quality on each frame. Frames with high detection confidence produce reliable component predictions; frames with low detection confidence produce unreliable predictions that confuse the decoder. A two-stage approach: first predict frame-level detection quality (e.g., using the max detection confidence across all 24 channels), then adjust the PSR decoder's behavior based on quality -- low-quality frames get more temporal smoothing, high-quality frames get less.

The detection quality predictor: Train a small MLP on backbone features to predict "detection reliability score" (correlation between detection confidence and prediction correctness). Frames where detection is confident and correct get high reliability; frames where detection is confident but wrong get low reliability. This learned reliability score adaptively controls the PSR decoder's confidence threshold and temporal smoothing strength.

**Context from detection-quality-gap analysis (118 Section 2, Anomaly 7):** PSR F1 tracks detection quality (mAP 0.208 -> 0.317 over epochs 8-11, F1 tracking proportionally). A detection quality predictor would identify frames where detection is likely wrong and apply PSR compensation.

**Question:** Would a two-stage PSR with learned detection quality prediction (3-layer MLP on backbone features predicting frame-level detection reliability, trained to minimize MSE between predicted reliability and actual detection correctness on validation set) improve PSR F1 from 0.144 to 0.22-0.35 by adaptively adjusting decoder confidence thresholds and temporal smoothing based on per-frame detection quality -- specifically, low-reliability frames (detection likely wrong) get higher confidence thresholds (fewer spurious transitions) and stronger temporal smoothing (more reliance on temporal context)?

**Why this matters:** Detection quality adaptive PSR is a novel concept that explicitly recognizes the bottleneck: PSR F1 is limited by detection quality. An adaptive decoder that degrades gracefully when detection quality is low provides better overall F1 than a fixed decoder. Estimated impact: +0.08-0.20 PSR F1.

**Constraints:** Code change to psr_transition.py (~60 lines for two-stage architecture). 25-epoch ablation on 3060. Must train the reliability predictor on held-out data (to avoid overfitting). Additional ~0.1M params for reliability MLP. Must define "detection correctness" ground truth (IoU > 0.5 with GT box of correct class).

**Hypothesis:** Two-stage quality-adaptive PSR achieves F1=0.22-0.35 at epoch 25. Low-quality frames (bottom 20% by detection confidence) show the largest F1 improvement (+0.10-0.20) because the adaptive behavior prevents spurious transitions on these frames. High-quality frames (top 20%) show minimal change. The reliability predictor achieves 70-80% accuracy at classifying frames as "reliable" vs "unreliable."

**Validation:** Implement detection quality predictor. Run 25-epoch ablation. Compare PSR F1, per-component F1 against fixed-decoder baseline. Analyze F1 by detection quality quintile. Success criterion: PSR F1 >= 0.22 at epoch 25.

---

### Q50. Multi-Decoder Ensemble: MonotonicDecoder + Transition Detector + Temporal Model

**Context:** Each PSR paradigm (fill-forward, transition detection, temporal) has different strengths: fill-forward excels at POS, transition detection excels at F1 timing, temporal models excel at consistency. A multi-decoder ensemble that combines all three paradigms, weighted by their per-component confidence, could achieve both high POS and high F1 simultaneously.

The ensemble mechanism: Three independent decoders process the same backbone features:
- D1: MonotonicDecoder (fill-forward, current) -- produces high POS
- D2: Transition detection head (Q46) -- produces high F1 timing
- D3: Temporal cross-attention (Q47) -- produces temporal consistency
The final prediction is a weighted average: P(c) = w1 * P1(c) + w2 * P2(c) + w3 * P3(c), where weights w1, w2, w3 are learned per component via a small gating network (2-layer MLP on global features, outputing 3 softmax weights per component).

**Context from model comparisons (114-comparability-vs-4-papers.md:280-295, 432-434):** B3 uses LSTM + transformer, STORM-PSR uses multi-modal temporal fusion. A multi-decoder ensemble is the closest analog to these SOTA approaches within our architectural constraints. Each decoder is weak individually, but their combination could approach SOTA.

**Question:** Would a three-decoder ensemble (MonotonicDecoder + Transition Detector + Temporal Cross-Attention, combined via learned per-component weights from a gating network) improve PSR F1 from 0.144 to 0.35-0.55 while maintaining POS above 0.95, by exploiting the complementary strengths of each paradigm -- and does the ensemble achieve better F1 than any single paradigm because the gating network learns to trust MonotonicDecoder for ordering (high POS) and transition detection for timing (high F1)?

**Why this matters:** A multi-decoder ensemble is the most technically ambitious PSR approach. If it achieves F1 > 0.35 with POS > 0.95, it would be the strongest PSR result on IndustReal. The ensemble approach demonstrates that the three paradigms are complementary rather than competing. This is publishable as a novel PSR architecture. Estimated impact: +0.20-0.40 PSR F1.

**Constraints:** Code changes for three-decoder architecture (~150 lines total). 50-epoch run on 5060 Ti. Adds ~3M params (transition head + temporal model + gating network). VRAM increase ~1.5 GB. Inference cost increases ~2x (three decoders). Must ensure training is stable (three decoders can compete, requiring careful loss balancing).

**Hypothesis:** Multi-decoder ensemble achieves PSR F1=0.35-0.55 and POS > 0.95 at epoch 50. The gating network assigns: w1 (MonotonicDecoder) = 0.4-0.6 (high POS), w2 (transition) = 0.2-0.4 (high F1), w3 (temporal) = 0.2-0.3 (consistency). Per-component weights show interpretable patterns: rare components (h4, h7-h10) get higher w2 (rely on transition detection for timing), common components (h0-h3) get higher w1 (rely on fill-forward for ordering).

**Validation:** Implement three-decoder ensemble with gating network. Run 50 epochs. Compare PSR F1, POS, edit against each individual decoder and against combined baseline. Success criterion: PSR F1 >= 0.35 with POS >= 0.95.

---

## Summary: Predicted Impact of All 50 Questions

| Category | Best Single Q Impact | Cumulative Potential | Key Metric |
|----------|---------------------|---------------------|-----------|
| 1 Architecture | +0.06-0.12 mAP50 | +0.15-0.30 mAP50 | Detection |
| 2 Training Recipe | +0.02-0.05 combined | +0.05-0.10 combined | Combined |
| 3 Loss Redesign | +0.10-0.25 PSR F1 | +0.20-0.50 PSR F1 | PSR F1 |
| 4 Data Strategy | +0.08-0.16 mAP50 | +0.15-0.30 mAP50 | Detection |
| 5 Multi-Task | +0.04-0.09 act F1 | +0.08-0.18 act F1 | Activity |
| 6 Cross-Dataset | +0.06-0.12 mAP50 | +0.12-0.24 mAP50 | Detection |
| 7 Detection-Specific | +0.06-0.12 mAP50 | +0.15-0.30 mAP50 | Detection |
| 8 Activity-Specific | +0.10-0.20 act F1 | +0.20-0.40 act F1 | Activity |
| 9 Ego-Pose-Specific | -0.8 to -1.6 deg MAE | -1.5 to -3.0 deg MAE | Pose |
| 10 PSR-Specific | +0.20-0.40 PSR F1 | +0.40-0.70 PSR F1 | PSR F1 |

## Selection Strategy for Maximum SOTA Impact

**Highest immediate impact (run first, inference-only):** Q42 Kalman filter (pose, -0.3 to -0.8 deg), Q48 hysteresis thresholds (PSR, +0.06-0.14 F1), Q41 6D rotation (pose, -0.8 to -1.6 deg with training). These require no training or at most 25-epoch ablations on the idle 3060.

**Highest long-term impact (requires full training):** Q35 multi-scale training (det, +0.03-0.07 mAP50_pc), Q46 transition detection (PSR, +0.20-0.40 F1), Q50 multi-decoder ensemble (PSR, +0.20-0.40 F1), Q21 CAGrad balancing (act, +0.04-0.09 F1), Q32 DINOv2 backbone (det, +0.03-0.08 mAP50). These require 50-epoch runs on the 5060 Ti.

**Most novel contribution (paper headline material):** Q15 multi-task NT-Xent (cross-task alignment), Q11 TAFL (transition-aware detection loss), Q30 universal assembly representation (cross-dataset metric learning), Q50 multi-decoder ensemble (complementary PSR paradigms). These are publishable as novel methods regardless of metric impact.

**Cost-benefit analysis for remaining GPU time (pre-AAIML submission):** The idle 3060 can run 25-epoch ablations in 2-3 days each. Priority: Q41 (6D pose, 2 days) > Q21 (CAGrad, 3 days) > Q34 (distillation, 3 days) > Q35 (multi-scale, 3 days) > Q12 (sequence contrastive PSR, 3 days). Total: ~14 3060-days for 5 experiments. The 5060 Ti (occupied with main run until ~July 16) then runs the best-combined configuration as a single 100-epoch run.

---

---

# Appendix A: Detailed Cross-Reference Matrix

Each question's primary metric target, secondary effects, and relationship to other questions. This matrix enables experiment sequencing where compatible improvements are batched together.

## Architecture (Q1-Q5) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q1 ConvNeXt-V2 FCMAE | det_mAP50 +0.06-0.12 | act_macro_F1 +0.01-0.02 | Q26 (any pretrain) | Q32 (DINOv2, incompatible backbones) |
| Q2 DyHead | combined +0.02-0.04 | act_macro_F1 +0.02-0.04 | Q21 (CAGrad, both improve activity gradient) | Q15 (NT-Xent, both modify shared features) |
| Q3 CST Neck | mAP50_pc +0.04-0.09 | PSR F1 +0.02-0.05 | Q5 (NAS-FPN, both are neck changes) | Q30 (neck-agnostic metric learning) |
| Q4 SimOTA Head | mAP50_pc +0.04-0.09 | mAP50 +0.04-0.08 | Q5 (OHEM removal, similar mechanism) | Q34 (distillation, different assignment signals) |
| Q5 NAS-FPN Neck | mAP50_pc +0.05-0.10 | PSR F1 +0.03-0.06 | Q3 (CST, combined could give +0.08-0.15) | Q32 (DINOv2, needs ConvNeXt features) |

## Training Recipe (Q6-Q10) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q6 Curriculum Learning | mAP50_pc +0.02-0.05 | mAP50 +0.01-0.03 | Q19 (adaptive sampling, same mechanism) | Q16 (tubelet sampling, different curriculum) |
| Q7 SAM Optimizer | combined +0.02-0.04 | all tasks +0.01-0.02 | Q8 (SWA, both find flat minima) | Q6 (curriculum, both modify optimization) |
| Q8 SWA | combined +0.02-0.05 | act_macro_F1 +0.01-0.03 | Q7 (SAM, combined flatter minima) | Q32 (EMA decay sweep, redundant) |
| Q9 LLRD | combined +0.02-0.04 | gradient conflict -0.2 cos | Q23 (GradVac, both reduce conflict) | Q7 (SAM, both modify optimizer) |
| Q10 Multi-Cycle LR | combined +0.02-0.05 | all tasks +0.01-0.02 | Q8 (SWA averaging over cycles) | Q6 (curriculum, different epoch schedules) |

## Loss Redesign (Q11-Q15) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q11 TAFL | mAP50_pc +0.04-0.09 | ch9-12 confusion -50% | Q4 (SimOTA, both improve rare classes) | Q2 (DyHead, modifies head design) |
| Q12 Seq Contrastive PSR | PSR F1 +0.10-0.25 | POS -0.01-0.02 | Q47 (temporal PSR, both temporal) | Q46 (transition detection, different paradigm) |
| Q13 Uncertainty Geodesic | fwd_MAE -1.0 to -1.6 deg | up_MAE -1.0 to -1.6 deg | Q12 (position removal, additive) | Q45 (MC Dropout, both model uncertainty) |
| Q14 Order-Reg PSR | PSR F1 +0.08-0.20 | POS -0.01-0.03 | Q48 (hysteresis, temporal consistency) | Q46 (transition detection, incompatible) |
| Q15 Multi-Task NT-Xent | combined +0.03-0.06 | grad cos > -0.1 | Q23 (GradVac, both align gradients) | Q2 (DyHead, both modify features) |

## Data Strategy (Q16-Q20) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q16 Tubelet Sampling | mAP50_pc +0.03-0.06 | PSR F1 +0.02-0.05 | Q18 (counterfactual, both add rare examples) | Q19 (adaptive sampling, different mechanism) |
| Q17 FixMatch SS | mAP50 +0.08-0.16 | mAP50_pc +0.07-0.14 | Q34 (distillation, both use YOLOv8m) | Q38 (pseudo-labels, similar mechanism) |
| Q18 Counterfactual Aug | mAP50_pc +0.05-0.11 | rare ch AP +0.08-0.20 | Q16 (tubelet, both target rare classes) | Q37 (synthetic Unity, different approach) |
| Q19 Adaptive Curriculum | mAP50_pc +0.02-0.05 | all classes more uniform | Q6 (curriculum, same direction) | Q16 (tubelet, fixed vs adaptive sampling) |
| Q20 Optical Flow | act_macro_F1 +0.04-0.09 | det_mAP50 +0.01-0.03 | Q36 (temporal head, both temporal) | Q40 (VideoMAE, redundant temporal signal) |

## Multi-Task Balancing (Q21-Q25) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q21 CAGrad | act_macro_F1 +0.04-0.09 | combined +0.01-0.02 | Q2 (DyHead, improved head + balance) | Q22 (GradNorm, different balance method) |
| Q22 GradNorm | act_macro_F1 +0.03-0.07 | det_mAP50 -0.01-0.02 | Q23 (IMTL-G, both are gradient methods) | Q21 (CAGrad, different optimization objective) |
| Q23 IMTL-G | act_macro_F1 +0.02-0.06 | combined +0.01-0.03 | Q15 (NT-Xent, gradient alignment) | Q21 (CAGrad, different math framework) |
| Q24 DWA | act_macro_F1 +0.03-0.07 | combined +0.01-0.02 | Q9 (LLRD, both adjust per-task learning) | Q21 (CAGrad, loss-based vs gradient-based) |
| Q25 GradVac | combined +0.02-0.05 | grad cos > -0.1 | Q15 (NT-Xent, both reduce gradient conflict) | Q22 (GradNorm, surgery vs magnitude balance) |

## Cross-Dataset Transfer (Q26-Q30) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q26 IKEA ASM Pretrain | det_mAP50 +0.06-0.12 | act_macro_F1 +0.03-0.07 | Q1 (FCMAE, combine pretrain sources) | Q17 (FixMatch, different supervision strategy) |
| Q27 Joint Training | det_mAP50 +0.03-0.08 | act_macro_F1 +0.03-0.07 | Q9 (LLRD, fine-tuning strategy) | Q26 (pretrain + finetune, same data, different method) |
| Q28 Domain Adversarial | mAP50_pc +0.03-0.08 | ch16/19/22 +0.05-0.15 | Q26 (IKEA ASM + domain adaptation) | Q15 (NT-Xent, both regularize features) |
| Q29 Cross-Dataset Pose | fwd_MAE -0.6 to -1.1 deg | up_MAE -0.6 to -1.1 deg | Q44 (IKEA pose aug, same data source) | Q13 (geodesic, different improvement mechanism) |
| Q30 Universal Assembly | mAP50_pc +0.04-0.08 | zero-shot transfer | Q27 (joint training, same multi-dataset approach) | Q26 (pretrain only, less ambitious) |

## Detection-Specific (Q31-Q35) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q31 Objects365 Pretrain | det_mAP50 +0.06-0.12 | mAP50_pc +0.05-0.10 | Q1 (FCMAE, combine pretrain sources) | Q32 (DINOv2, different backbone architecture) |
| Q32 DINOv2 Backbone | det_mAP50 +0.03-0.08 | act_macro_F1 +0.04-0.09 | Q35 (multi-scale, scale-invariant DINOv2) | Q1 (ConvNeXt-V2, incompatible backbone family) |
| Q33 ConvNeXt-Nano | det_mAP50 -0.01 to -0.03 | speed 1.5x, params -46% | Q7 (SAM, flatter minima on smaller model) | Q31 (Objects365, less capacity for pretrain) |
| Q34 YOLOv8m Distill | det_mAP50 +0.03-0.08 | distills knowledge cheaply | Q17 (FixMatch, both leverage YOLOv8m) | Q4 (SimOTA, different label assignment) |
| Q35 Multi-Scale Train | mAP50_pc +0.03-0.07 | small ch +0.05-0.15 | Q32 (DINOv2, scale-invariant features) | Q33 (Nano, smaller model less effective) |

## Activity-Specific (Q36-Q40) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q36 Hierarchical TCN | temporal F1 +0.07-0.13 | all actions more uniform | Q20 (optical flow, both temporal signal) | Q40 (VideoMAE, different temporal approach) |
| Q37 Verb-Noun Hierarchy | macro-F1 +0.04-0.09 | pred_distinct +15 | Q9 (LLRD, hierarchical features) | Q40 (VideoMAE, dual backbone complexity) |
| Q38 Detection-Aug Act | macro-F1 +0.03-0.08 | tighten/loosen +0.05-0.10 | Q4 (SimOTA, better detection features) | Q20 (flow, different auxiliary signal) |
| Q39 SMOTE Oversampling | macro-F1 +0.02-0.06 | rare classes +0.03-0.10 | Q19 (adaptive sampling, both target rare) | Q6 (curriculum, different rare-class strategy) |
| Q40 VideoMAE Dual-Back | temporal F1 +0.10-0.20 | matches MViTv2 remapped | Q36 (hierarchical TCN + VideoMAE) | Q32 (DINOv2, backbone incompatibility) |

## Ego-Pose-Specific (Q41-Q45) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q41 6D Rotation | fwd_MAE -0.8 to -1.6 deg | up_MAE -0.8 to -1.6 deg | Q13 (uncertainty geodesic, loss improvement) | Q12 (position removal, orthogonal improvements) |
| Q42 Kalman Filter | fwd_MAE -0.3 to -0.8 deg | up_MAE -0.3 to -0.6 deg | Q41 (6D + Kalman additive, ~1.0-2.2 deg total) | Q45 (MC Dropout, both smoothing but different) |
| Q43 Multi-Scale Pose | fwd_MAE -0.3 to -0.8 deg | up_MAE -0.3 to -0.6 deg | Q13 (uncertainty + multi-scale additive) | Q42 (Kalman, both temporal but architectural) |
| Q44 IKEA Pose Aug | fwd_MAE -0.6 to -1.1 deg | up_MAE -0.6 to -1.1 deg | Q29 (IKEA pre-train, both cross-dataset pose) | Q14 (random rotation, different aug approach) |
| Q45 MC Dropout | fwd_MAE -0.1 to -0.6 deg | uncertainty calibration | Q13 (uncertainty + MC dropout, both model uncertainty) | Q42 (Kalman, averaging vs filtering) |

## PSR-Specific (Q46-Q50) Cross-References

| Question | Primary Metric | Secondary Metric | Synergistic With | Antagonistic With |
|----------|---------------|-----------------|-----------------|------------------|
| Q46 Transition Detection | PSR F1 +0.20-0.40 | POS -0.04-0.08 | Q50 (ensemble, component of larger model) | Q14 (order-reg, incompatible paradigms) |
| Q47 Temporal Cross-Attn | PSR F1 +0.10-0.25 | rare ch tau -2 to -4 frames | Q12 (contrastive, both temporal models) | Q46 (transition detection, different temporal model) |
| Q48 Hysteresis Threshold | PSR F1 +0.06-0.14 | POS -0.01-0.02 | Q14 (order-reg + hysteresis, regularize + threshold) | Q46 (transition detection, different paradigm) |
| Q49 Quality-Adaptive PSR | PSR F1 +0.08-0.20 | adaptive decoding | Q47 (temporal + quality-adaptive) | Q48 (hysteresis, different threshold method) |
| Q50 Multi-Decoder | PSR F1 +0.20-0.40 | POS > 0.95 | Q46+47+48 (all three subsumed into ensemble) | Q46 (single transition head, less complex) |

---

# Appendix B: Implementation Complexity and GPU Budget

| Question | Code Complexity | GPU Hours | Run Type | Priority | Dependencies |
|----------|---------------|-----------|----------|----------|-------------|
| Q1 FCMAE | Low (config) | 160h (50ep) | Full train | Medium | timm weights |
| Q2 DyHead | High (150 lines) | 160h (50ep) | Full train | Medium | None |
| Q3 CST Neck | High (150 lines) | 160h (50ep) | Full train | Low | None |
| Q4 SimOTA Head | High (200 lines) | 160h (50ep) | Full train | Medium | None |
| Q5 NAS-FPN Neck | High (100 lines) | 160h (50ep) | Full train | Low | None |
| Q6 Curriculum | Medium (50 lines) | 160h (100ep) | Full train | Medium | None |
| Q7 SAM | Medium (30 lines) | 320h (50ep double) | Full train | Low | SAM library |
| Q8 SWA | Low (20 lines) | 1h (post-hoc) | Inference | High | None |
| Q9 LLRD | Low (30 lines) | 160h (50ep) | Full train | Medium | None |
| Q10 Multi-Cycle | Low (10 lines) | 160h (100ep) | Full train | Low | None |
| Q11 TAFL | Medium (30 lines) | 160h (50ep) | Full train | High | Assembly manual |
| Q12 Seq Contrast | High (80 lines) | 160h (50ep) | Full train | High | None |
| Q13 Uncertainty Geo | Low (20 lines) | 80h (25ep 3060) | Ablation | High | None |
| Q14 Order-Reg PSR | Medium (40 lines) | 80h (25ep 3060) | Ablation | High | Assembly manual |
| Q15 NT-Xent | High (60 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q16 Tubelet | Medium (40 lines) | 160h (50ep) | Full train | Medium | None |
| Q17 FixMatch | High (150 lines) | 160h (50ep) | Full train | High | None |
| Q18 Counterfactual | High (120 lines) | 10h CPU + 160h GPU | Generation + train | Medium | Inpainting library |
| Q19 Adaptive Curr | Low (30 lines) | 160h (50ep) | Full train | Low | None |
| Q20 Optical Flow | High (50 lines) | 50h flow + 160h train | Preprocess + train | Medium | FlowNet2 |
| Q21 CAGrad | Medium (30 lines) | 80h (25ep 3060) | Ablation | High | CAGrad library |
| Q22 GradNorm | Medium (50 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q23 IMTL-G | Low (20 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q24 DWA | Low (15 lines) | 80h (25ep 3060) | Ablation | Low | None |
| Q25 GradVac | Medium (40 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q26 IKEA Pretrain | Medium (dataset) | 50h pre + 160h fine | Two-stage train | Low | IKEA ASM dataset |
| Q27 Joint Train | High (100 lines) | 240h (50ep joint) | Full train | Low | IKEA ASM dataset |
| Q28 Domain Advers | Medium (50 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q29 Cross Pose | Medium (dataset) | 40h pre + 40h fine | Two-stage train | Low | IKEA ASM pose |
| Q30 Universal Metric | High (150 lines) | 400h (75ep) | Full train | Very Low | IKEA ASM dataset |
| Q31 Objects365 | Low (config) | 160h (50ep) | Full train | High | Objects365 weights |
| Q32 DINOv2 Backbone | High (80 lines) | 160h (50ep) | Full train | Medium | DINOv2 weights |
| Q33 ConvNeXt-Nano | Low (config) | 160h (150ep) | Full train | Low | timm weights |
| Q34 YOLOv8m Distill | Medium (30 lines) | 4h pred + 160h train | Preprocess + train | High | D1 results |
| Q35 Multi-Scale Train | Low (10 lines) | 200h (50ep larger) | Full train | High | None |
| Q36 Hierarchical TCN | High (80 lines) | 160h (50ep) | Full train | Medium | None |
| Q37 Verb-Noun Head | Medium (60 lines) | 160h (50ep) | Full train | Medium | Taxonomy mapping |
| Q38 Detection-Aug Act | Low (15 lines) | 80h (25ep 3060) | Ablation | High | None |
| Q39 SMOTE | Medium (30 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q40 VideoMAE Dual | High (100 lines) | 240h (50ep) | Full train | Low | VideoMAE weights |
| Q41 6D Rotation | Medium (30 lines) | 80h (25ep 3060) | Ablation | High | None |
| Q42 Kalman Filter | Low (20 lines) | 1h (inference) | Inference | Immediate | None |
| Q43 Multi-Scale Pose | Medium (40 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q44 IKEA Pose Aug | Medium (50 lines) | 24h pre + 80h train | Preprocess + train | Medium | IKEA ASM dataset |
| Q45 MC Dropout | Low (20 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q46 Transition Det | High (100 lines) | 160h (50ep) | Full train | High | None |
| Q47 Temporal Cross-Attn | High (80 lines) | 160h (50ep) | Full train | Medium | None |
| Q48 Hysteresis | Low (20 lines) | 1h (inference) | Inference | Immediate | None |
| Q49 Quality-Adaptive | High (60 lines) | 80h (25ep 3060) | Ablation | Medium | None |
| Q50 Multi-Decoder | Very High (150 lines) | 160h (50ep) | Full train | Low | Q46+47+48 results |

---

# Appendix C: Recommended Experiment Sequencing

The recommended execution sequence maximizes information gain per GPU-hour, with early decisions de-risking later investments.

## Phase 0: Immediate Inference-Only (0 GPU hours, 4 hours engineer time)

Run these today on the idle 3060 before anything else:

1. **Q42 Kalman Filter for pose** (1 hour, -0.3 to -0.8 deg MAE improvement)
2. **Q48 Hysteresis thresholding for PSR** (1 hour, +0.06-0.14 PSR F1 improvement)
3. **Q8 SWA post-hoc averaging** (1 hour on completed main run, +0.02-0.05 combined)
4. **Q34 YOLOv8m distillation prep** (1 hour to run YOLOv8m on training frames via D1)

These four experiments cost zero training time, use existing checkpoints, and provide immediate metric improvements that strengthen the paper without delaying any other work.

## Phase 1: 3060 Ablations (25 epochs each, ~10 GPU days total)

Run these on the idle 3060 while the main run completes on the 5060 Ti:

1. **Q41 6D rotation representation** (3 days, -0.8 to -1.6 deg MAE, de-risks entire pose contribution)
2. **Q21 CAGrad multi-task balancing** (3 days, +0.04-0.09 act F1, de-risks activity contribution)
3. **Q38 Detection-augmented activity head** (3 days, +0.03-0.08 act F1, cheap cross-task test)
4. **Q35 Multi-scale training** (3 days, +0.03-0.07 mAP50_pc, standard detection improvement)

These four experiments each target a different metric, can be evaluated independently, and provide data for the ablation table of the paper.

## Phase 2: 3060 + 5060 Ti Parallel (25-50 epochs, ~20 GPU days total)

After Phase 1 completes (approximately July 10-12), run the best-improved configuration from Phase 1 as a full 100-epoch run on the 5060 Ti, while the 3060 tests higher-risk experiments:

5060 Ti: **Integrated best combination** (CAGrad + 6D pose + multi-scale train + detection-augmented act) for 100 epochs.

3060: Test higher-risk, higher-reward experiments:
1. **Q12 Sequence-level contrastive PSR** (3 days, +0.10-0.25 PSR F1, highest potential PSR impact)
2. **Q11 Transition-aware focal loss** (3 days, +0.04-0.09 mAP50_pc, novel method)
3. **Q31 Objects365 pretraining** (3 days, +0.06-0.12 mAP50, highest detection potential)

## Phase 3: Full Assembly (Post-July 16, after main run completes)

Take the Phase 2 integrated best checkpoint and apply:
1. Q42 Kalman filter (inference, 1 hour)
2. Q48 hysteresis thresholds (inference, 1 hour)
3. Q8 SWA averaging (inference, 1 hour)

Evaluate the final model on the full 38K validation set (D3). Report all metrics with both the combined model and per-ablation contributions clearly decomposed in the paper.

## SOTA Target After Phases 0-3

| Metric | Current Best | Post-Phase 0 (Inference) | Post-Phase 1 (3060 Ablations) | Post-Phase 3 (Full Integrated) |
|--------|-------------|------------------------|------------------------------|-------------------------------|
| det_mAP50 | 0.317 | 0.32-0.34 | 0.35-0.40 | 0.42-0.50 |
| det_mAP50_pc | 0.506 | 0.51-0.53 | 0.54-0.58 | 0.58-0.65 |
| act_macro_F1 | 0.110 | 0.11-0.12 | 0.14-0.18 | 0.16-0.22 |
| pose_fwd_MAE | 8.14 deg | 7.3-7.8 deg | 6.5-7.0 deg | 6.0-6.8 deg |
| pose_up_MAE | 7.06 deg | 6.5-6.8 deg | 5.5-6.0 deg | 5.0-5.8 deg |
| PSR_POS | 0.968 | 0.96-0.97 | 0.95-0.97 | 0.94-0.97 |
| PSR_F1 | 0.144 | 0.20-0.28 | 0.25-0.40 | 0.35-0.55 |
| combined | 0.363 | 0.37-0.39 | 0.39-0.44 | 0.42-0.50 |

These targets represent 40-70% closure of the detection gap (vs YOLOv8m 0.838), 50-80% of the activity gap (vs remapped MViTv2 0.25-0.35), 15-35% closure of the PSR F1 gap (vs STORM-PSR 0.901), and pose beating all existing baselines toward the HL2 noise floor. The combined improvement across all metrics maps to a 40-60% better position in the AAIML review process compared to the current numbers.

---

---

# Appendix D: Detailed Validation Protocol for Each Question

Each experiment must follow a standardized validation protocol to ensure results are comparable and reproducible. This appendix defines the shared protocol and the per-question variations.

## Shared Validation Framework

All experiments share:
- **Checkpoint baseline:** epoch 11 of the current main run (stage_rf4, as documented in 111-overview.md). All ablations compare against this epoch 11 baseline unless otherwise noted.
- **Validation set:** 38K labeled frames (EVAL_MAX_BATCHES=0 for full validation, 250 batches for rapid iteration). D3 full validation is the final evaluation for all reported numbers.
- **Metrics recorded:** det_mAP50, det_mAP50_pc, per-class AP (all 24 channels), act_macro_F1, act_Top1, act_Top5, pose_fwd_MAE, pose_up_MAE, PSR_POS, PSR_F1, PSR_edit, PSR_comp_acc, combined metric.
- **Training config for ablations:** BATCH_SIZE=4, NUM_WORKERS=0 (3060 compatibility), loss weights and Kendall parameters from baseline config, 25 epochs unless noted. Fresh run from random init unless pretrained.
- **Success criterion threshold:** Each question defines a primary success criterion. If the primary criterion is met, the experiment is successful. If not, the experiment is negative but still informative (rules out that approach).

## Per-Question Protocol Variations

### Q1 Validation Addendum
- Compare ConvNeXt-V2 FCMAE pretrained epoch 25 vs ConvNeXt-V1 random init epoch 25
- Control: same training recipe except pretraining
- Diagnostic: compute per-channel AP difference, focus on channels 16/19/22
- If success criterion not met (<0.38 mAP50): analyze whether features collapse (channel AP across the board lower vs baseline)

### Q2 Validation Addendum
- Compare DyHead combined metric epoch 25 vs per-task head baseline epoch 25
- Diagnostic: compute per-task gradient shares after DyHead insertion (gradient logging at epoch 5, 15, 25)
- If activity does not improve: DyHead's unified attention may favor dominant tasks (detection, PSR) over minority tasks (activity) -- check attention weight distribution

### Q3 Validation Addendum
- Compare CST neck mAP50_pc epoch 50 vs FPN baseline epoch 50
- Diagnostic: extract cross-scale attention weights and check which scale pairs have highest attention for the 9-12 confusion cluster vs high-AP channels
- If confusion channels do not improve: CST's cross-scale attention may still receive insufficient gradient from detection head -- check per-layer gradient norms

### Q4 Validation Addendum
- Compare SimOTA head mAP50 epoch 25 vs RetinaNet head + OHEM baseline epoch 25
- Diagnostic: compute per-class positive anchor counts under SimOTA vs OHEM assignment
- If rare channels do not improve: SimOTA may assign zero positive anchors to classes with very low AP because the cost function (cls loss + reg loss) still favors common classes

### Q5 Validation Addendum
- Compare NAS-FPN neck mAP50_pc epoch 50 vs FPN baseline epoch 50
- Diagnostic: visualize learned routing weights to identify which cross-scale connections are most utilized
- If mAP50_pc < 0.56: NAS-FPN may overfit the 26K training frames with its additional capacity

### Q6 Validation Addendum
- Compare curriculum schedule epoch 100 vs uniform sampling epoch 100
- Diagnostic: track per-class AP trajectory separately for easy/medium/hard groups
- If hard channels do not improve: curriculum ordering may be incorrect -- verify the difficulty ordering matches actual learnability (not just GT count)

### Q7 Validation Addendum
- Compare SAM (rho=0.05) epoch 50 (100 compute) vs AdamW baseline epoch 50 (50 compute)
- Diagnostic: measure loss landscape sharpness (Hessian eigenvalue estimate via power iteration) at convergence
- If SAM fails to improve: rho=0.05 may be too large for multi-task setting -- try rho sweep {0.01, 0.02, 0.05, 0.1} in 5-epoch probes

### Q8 Validation Addendum
- Compare SWA (epochs 75-100 avg) vs best single EMA checkpoint
- Diagnostic: compute per-task loss at the SWA minimum and compare sharpness (Hessian)
- SWA is free to run -- always apply as final step regardless of outcome

### Q9 Validation Addendum
- Sweep LLRD decay {0.85, 0.90, 0.95, 0.99 (=baseline)} in 15-epoch probes on 3060
- Compare combined metric and per-task metrics at epoch 15 for each decay
- Diagnostic: compute per-stage gradient norms to verify LLRD is effectively controlling learning speed

### Q10 Validation Addendum
- Compare 2-cycle schedule epoch 100 vs single-cycle baseline epoch 100
- Diagnostic: track combined metric trajectory through both cycles -- cycle 2 should improve upon cycle 1's best
- If cycle 2 does not improve: the peak LR in cycle 2 (0.5x) may be too high and destabilize -- reduce to 0.25x

### Q11 Validation Addendum
- Compare TAFL epoch 50 vs FocalLoss baseline epoch 50
- Diagnostic: compute 24x24 confusion matrix for both models, measure assembly-violation rate
- If confusion cluster (ch9-12) does not improve: transition cost matrix weights may be wrong -- verify Hamming distances against assembly manual and adjust penalty for assembly-logic violations

### Q12 Validation Addendum
- Sweep contrastive lambda {0.1, 0.3, 0.5} in 25-epoch probes on 3060
- Compare PSR F1 and POS for each lambda vs no-contrastive baseline
- Diagnostic: compute t-SNE visualization of sequence-level features -- positive pairs should cluster, negative pairs should separate
- If F1 < 0.25: temperature may be too high/low, or negative sampling strategy may be creating false negatives

### Q13 Validation Addendum
- Compare uncertainty-weighted geodesic epoch 25 vs standard geodesic epoch 25
- Diagnostic: compute per-sample uncertainty (sigma) and correlation with prediction error
- If improvement < 0.5 deg MAE: the learned variance may not correlate with error -- plot sigma vs error scatter

### Q14 Validation Addendum
- Sweep order-regularization lambda {0.1, 0.3, 0.5, 1.0} in 25-epoch probes on 3060
- Compare PSR F1 and POS for each lambda
- Diagnostic: compute per-component tau (detection delay) to measure cascading delay reduction
- If lambda > 0.3 degrades POS: order regularization is too strong and forcing incorrect ordering on rare components

### Q15 Validation Addendum
- Sweep temperature {0.1, 0.5, 1.0} x lambda {0.05, 0.1, 0.2} in 10-epoch probes on 3060
- Compute grad_cosine_probe (F12) at epoch 10 for each combination
- Compare combined metric against baseline for best configuration at epoch 25
- If NT-Xent fails: the two-augmentation requirement doubles compute and may not fit in 16 GB at B=4 -- test at B=2

### Q16 Validation Addendum
- Sweep tubelet alpha {1, 2, 5, 10} in 25-epoch probes on 3060
- Compare per-class AP for transitional channels (16-23) vs uniform baseline
- If channels 16/19/22 remain < 0.05: the transitional states may be fundamentally hard (occlusion, viewpoint) and need architectural improvements, not just more data

### Q17 Validation Addendum
- Sweep confidence threshold {0.7, 0.8, 0.9, 0.95} in 15-epoch probes on 3060
- Compare mAP50 and pseudo-label quality (precision of pseudo-labels vs held-out GT)
- If mAP50 < 0.40: the pseudo-label quality may be too low -- check precision/recall of YOLOv8m predictions on frames without GT

### Q18 Validation Addendum
- Generate counterfactual images for 10 target channels with 10x oversampling
- Compare mAP50_pc with counterfactual augmentation vs baseline at epoch 50
- If counterfactual quality is poor (visible artifacts): use LaMa inpainting or partial-conv inpainting for better results

### Q19 Validation Addendum
- Compare learning-progress curriculum epoch 50 vs uniform baseline epoch 50
- Diagnostic: plot sampling weights over time -- they should decrease for high-AP channels and increase for low-AP channels
- If no improvement: the learning-progress metric (target_AP - current_AP) may be noisy -- smooth over 10 epochs

### Q20 Validation Addendum
- Precompute FlowNet2 optical flow for training set
- Compare act_macro_F1 with auxiliary flow loss epoch 50 vs RGB-only baseline epoch 50
- If flow does not improve activity: the flow features may be too noisy for the activity head to use -- try using flow as additional input channel (concatenate) rather than auxiliary prediction task

### Q21 Validation Addendum
- Sweep CAGrad alpha {0.4, 0.5, 0.6} in 10-epoch probes on 3060
- Compare gradient composition and per-task metrics for best alpha vs Kendall baseline at epoch 25
- If activity does not improve > 0.01: CAGrad may be over-constrained by detection/PSR gradient directions -- check per-step gradient similarity

### Q22 Validation Addendum
- Sweep adaptive alpha decay schedule {linear, exponential, step} in 10-epoch probes
- Compare gradient composition evolution over 25 epochs
- If activity does not improve: GradNorm may destablize the HP_PREC_CAP mechanism -- check pose gradient after GradNorm

### Q23 Validation Addendum
- Compare IMTL-G epoch 25 vs Kendall baseline epoch 25
- Diagnostic: compute gradient direction alignment (cosine similarity) for all 6 task pairs
- If no improvement: IMTL-G equal projection may give activity too much influence, causing detection to degrade -- check detection mAP50 for degradation > 0.02

### Q24 Validation Addendum
- Sweep DWA temperatures {1.0, 2.0, 4.0} for each task in 10-epoch probes
- Compare loss-rate ratios to verify DWA correctly identifies slow-improving tasks
- If no improvement: DWA's loss-based weighting may be too slow to respond to gradient needs -- try updating every epoch instead of every 5

### Q25 Validation Addendum
- Sweep GradVac target similarity {-0.1, 0.0, 0.2, 0.4} in 10-epoch probes
- Compute cosine similarity trajectories for all task pairs
- If no improvement: target_sim=0.2 may be too aggressive -- det and pose may genuinely be conflicting tasks with negative cosine, and forcing them toward positive similarity may harm both

### Q26 Validation Addendum
- Pre-train on IKEA ASM for 50 epochs (use IKEA ASM state labels)
- Fine-tune on IndustReal for 50 epochs
- Compare against random-init baseline at fine-tuning epochs 5, 10, 25, 50
- If improvement < 0.03 mAP50: the IKEA ASM and IndustReal assembly taxonomies may be too different for effective transfer

### Q27 Validation Addendum
- Jointly train on IndustReal (4 tasks) + IKEA ASM (state detection + action detection)
- Alternating batches (50% IndustReal, 50% IKEA ASM)
- Compare IndustReal metrics at epoch 50 vs single-dataset baseline
- If IKEA ASM hurts IndustReal performance: the different image statistics (furniture vs industrial setting) may be interfering -- try lower IKEA ASM sampling ratio (25%)

### Q28 Validation Addendum
- Define viewpoint domains by participant ID (12 domains)
- Sweep gradient reversal lambda {0.01, 0.1, 0.5} in 10-epoch probes
- Measure domain classifier accuracy vs baseline
- If transitional channels (16/19/22) do not improve: their failure may be due to feature learning, not viewpoint specificity -- compare with full-data baseline

### Q29 Validation Addendum
- Pre-train pose branch on IKEA ASM head pose data (25 epochs, frozen backbone)
- Fine-tune on IndustReal pose (25 epochs)
- Compare forward/up MAE at each fine-tuning epoch against scratch-trained baseline
- If MAE reduction < 0.5 deg: IKEA ASM head coordinate frames may be incompatible with HL2 -- check for systematic bias

### Q30 Validation Addendum
- Define component correspondence between IKEA ASM and IndustReal
- Train metric learning on both datasets jointly (50 epochs)
- Evaluate zero-shot: train on IKEA ASM + 10% IndustReal, test on held-out 90% IndustReal components
- If zero-shot fails: component-bit features may not transfer -- try fine-grained metric learning on shared visual features (object presence) rather than component identity

### Q31 Validation Addendum
- Download Objects365-pretrained ConvNeXt-Tiny weights (via timm or detectron2)
- Fine-tune with discriminative LR (backbone frozen 5 epochs at 1e-5, then 5e-5)
- Compare det_mAP50 epoch 25 vs random-init epoch 25
- If improvement < 0.05: Objects365 features may be too generic for assembly-specific detection

### Q32 Validation Addendum
- Integrate DINOv2-S as frozen backbone (ViT-S, 21M params)
- Train only task heads for 50 epochs
- Compare all metrics vs ConvNeXt-Tiny random init at epoch 25
- If detection does not improve: the frozen ViT features may lack the spatial resolution needed for small component detection -- try fine-tuning ViT at low LR (1e-6) for last 2 blocks

### Q33 Validation Addendum
- Train ConvNeXt-Nano for 150 epochs (same compute budget as Tiny 100 epochs)
- Compare compute-normalized metrics (Nano epoch 150 vs Tiny epoch 100)
- If Nano is > 0.03 worse than Tiny: the parameter reduction may be too aggressive for 4-task learning

### Q34 Validation Addendum
- Run YOLOv8m on training frames (via D1)
- Sweep distillation lambda {0.1, 0.5, 1.0, 2.0} and temperature {3, 5, 10}
- Compare det_mAP50 at epoch 25 for best config vs non-distilled baseline
- If improvement < 0.03: YOLOv8m logits may be too similar to our own (converged to similar solution) -- check per-class KL divergence

### Q35 Validation Addendum
- Implement random scale sampling {160, 192, 224, 256, 288}
- Train 50 epochs with multi-scale
- Compare mAP50_pc at epoch 50 vs fixed 224 baseline
- If mAP50_pc improvement < 0.02: the scale variation in ASD may be minimal (all components are at similar distance) -- check scale distribution in validation set

### Q36 Validation Addendum
- Implement dilated TCN (dilations 1,2,4,8) with 2-level hierarchical pooling
- Train as temporal activity head on 16-frame clips (T2 architecture)
- Compare temporal macro-F1 at epoch 50 vs 2-layer TCN baseline
- If temporal F1 < 0.20: the hierarchical pooling may lose temporal precision needed for short actions -- try stride=2 pooling instead of stride=4

### Q37 Validation Addendum
- Implement verb branch (7 classes) + noun branch (20+ classes) + bilinear combination
- Train 50 epochs with both auxiliary and final 69-class losses
- Compare macro-F1 vs flat 69-class baseline at epoch 50
- If verb branch accuracy < 60%: the 7 verb classes may be too coarse (check inter-verb confusion matrix)

### Q38 Validation Addendum
- Concatenate 24 detection logits (stop-grad) to backbone features for activity head
- Compare macro-F1 at epoch 25 vs standard activity head
- Diagnostic: check if activity head learns to use detection logits (visualize weights on detection input)
- If no improvement: detection logits may be too noisy for current detection quality (mAP=0.317) -- try using only top-5 confident detection classes

### Q39 Validation Addendum
- Implement SMOTE on backbone features for bottom-30 activity classes
- Sweep SMOTE ratio {2x, 5x, 10x} in 15-epoch probes
- Compare macro-F1 vs no-SMOTE baseline at epoch 15
- If macro-F1 improvement < 0.02: SMOTE on features may produce unrealistic combinations -- try SMOTE on raw features + label smoothing

### Q40 Validation Addendum
- Download VideoMAE-pretrained ViT-T weights (5.7M params)
- Implement dual-backbone: VideoMAE for 16-frame clips (activity) + ConvNeXt-Tiny for single frames (det/pose/PSR)
- Late fusion: concatenate VideoMAE clip features with ConvNeXt per-frame features
- Compare temporal macro-F1 against single-backbone temporal head at epoch 50
- If temporal F1 < 0.20: late fusion may be suboptimal -- try cross-attention fusion between the two backbone features

### Q41 Validation Addendum
- Implement 6D rotation head (output 3x2 matrix, complete to 3x3 via Gram-Schmidt)
- Use geodesic loss on reconstructed rotation matrices
- Compare forward/up MAE at epoch 25 vs 3D vector MSE (current) and 3D vector geodesic (Q11)
- If improvement < 0.5 deg over Q11: the 6D representation may not be fully utilized by the current MLP head -- try adding one more hidden layer

### Q42 Validation Addendum
- Implement Kalman filter on per-frame forward/up predictions
- Tune Q and R on validation set (grid search: Q in {1e-5, 1e-4, 1e-3}, R in {1e-3, 1e-2, 1e-1})
- Compare smoothed vs raw MAE on epoch 11 checkpoint
- If MAE reduction < 0.2 deg: per-frame noise may already be low -- check prediction variance across consecutive frames

### Q43 Validation Addendum
- Implement coarse-to-fine pose head (P4 -> P2 refinement)
- Sweep coarse pose dim {3, 6, 9} in 10-epoch probes
- Compare forward MAE at epoch 25 vs single-scale baseline
- If no improvement: P2 features may be too noisy for pose regression -- try using P3 instead of P2 for refinement

### Q44 Validation Addendum
- Pre-compute IKEA ASM pose trajectories (forward/up across sequences)
- For each IndustReal frame, randomly replace pose with nearest IKEA ASM pose
- Compare forward MAE at epoch 25 vs no-aug baseline and random rotation baseline
- If no improvement: the IKEA ASM head kinematics may not be compatible (different camera mounting) -- check distribution shift

### Q45 Validation Addendum
- Add dropout (rate=0.1) to pose MLP
- Run 20 stochastic forward passes at inference
- Compare MC average MAE vs point estimate and compute ECE
- If ECE > 0.2: the dropout uncertainty may not be well-calibrated -- try temperature scaling for calibration

### Q46 Validation Addendum
- Implement transition detection head (predict transition frames directly)
- Compute transition labels from presence labels (diff over frames)
- Compare PSR F1 and POS at epoch 50 vs MonotonicDecoder baseline
- If transition detection F1 < 0.35: the transition labels may be too sparse (few frames have transitions) -- try with class-balancing loss

### Q47 Validation Addendum
- Implement 32-frame transformer encoder with cross-attention
- Apply fill-forward constraint on transformer outputs
- Compare PSR F1 at epoch 50 vs per-frame baseline
- If F1 improvement < 0.10: 32-frame window may be too short for long assembly sequences -- try 64-frame window

### Q48 Validation Addendum
- Grid-search tau_high {0.4, 0.5, 0.6, 0.7} x tau_low {0.1, 0.2, 0.3, 0.4}
- Evaluate on held-out portion of validation
- Compare F1 and POS against single-threshold baseline
- Grid search is exhaustive (16 combinations) but fast (inference only, < 30 min for all)

### Q49 Validation Addendum
- Train detection reliability predictor on held-out frames
- Implement adaptive decoder that adjusts thresholds based on reliability
- Compare PSR F1 at epoch 25 vs fixed-decoder baseline
- If no improvement: reliability predictor may not generalize -- try using simpler heuristic (detection max confidence quantile) instead of learned predictor

### Q50 Validation Addendum
- Implement three-decoder architecture with gating network
- Train all decoders jointly with shared backbone
- Compare PSR F1 and POS at epoch 50 vs individual decoders and all pairs
- If gating network collapses to one decoder: the gating loss may need entropy regularization (encourage using all decoders)

---

# Appendix E: Negative Result Analysis Template

Every experiment in this document is designed to be informative regardless of outcome. This appendix provides the standardized template for analyzing negative results (experiments that fail to meet their success criterion). Each negative result must be analyzed using this template before being discarded, as the analysis may reveal fundamental insights that reshape the paper's narrative.

## Negative Result: The experiment failed to meet the primary success criterion.

**1. Technical Verification: Did the experiment actually run as intended?**
- Check training logs for anomalies (loss spikes, NaN gradients, LR deviations)
- Verify checkpoint integrity (load best checkpoint, verify loss landscape)
- Confirm code changes were active (assertion-based verification at runtime)
- If code change was supposed to modify model behavior, did it? (log magnitude of change)

**2. Statistical Analysis: How confident are we that the negative result is real?**
- Compute effect size with 95% confidence interval
- If confidence interval overlaps zero, the result is inconclusive (need more training or lower variance)
- Run a second seed to rule out seed-specific effects

**3. Mechanism Diagnosis: Why did the experiment fail?**
- Proposed mechanism: [what was supposed to happen]
- Observed mechanism: [what actually happened based on diagnostics]
- Gap analysis: [specific diagnostic that reveals where the mechanism broke down]
- Example diagnoses: gradient not flowing through new component, activation collapse, numerical instability, competing effect from another part of the model

**4. Refinement Possibility: Can the experiment be modified to succeed?**
- Hyperparameter sensitivity: is there a region of hyperparameter space that was not explored?
- Architectural compatibility: does the change conflict with another architectural component?
- Scale requirement: does the change need more compute, more data, or different initialization?

**5. Paper Impact Assessment: What does this negative result mean for the paper narrative?**
- If the hypothesis was central to a contribution claim, does the negative result require modifying the claim?
- Can the negative result itself be published as an insight? (e.g., "We found that X does not transfer to assembly tasks, contrary to prior work on Y")
- Does the negative result invalidate any published numbers? (if yes, immediate action required)

## Example: Negative Result for Q21 (CAGrad)

If CAGrad fails to improve activity macro-F1 beyond 0.110:

1. **Technical verification:** Check CAGrad gradient logs -- is the inner optimization solving correctly? Verify alpha parameter was applied correctly.
2. **Statistical analysis:** Run a second seed (seed 7). If both seeds show no improvement, the negative result is robust.
3. **Mechanism diagnosis:** If activity gradient share did not increase (remained ~14.8%), CAGrad may have been dominated by PSR/detection gradients that overwhelm the CAGrad optimization objective. The "worst-case" task improvement rate may still be determined by detection/PSR because their gradients are much larger.
4. **Refinement:** Try CAGrad with per-task gradient clipping (equalize magnitudes before CAGrad), or try different alpha values (lower alpha gives more weight to worst-case task).
5. **Paper impact:** The hypothesis "CAGrad improves activity" is not supported. The paper's multi-task balancing section shifts from "CAGrad outperforms Kendall" to "Kendall remains competitive with gradient surgery methods on this 4-task problem" -- still publishable but weaker.

---

# Appendix F: Long-Tail Experiment Tracking

This document contains 50 experiments. A tracking system is needed to manage their execution. Each experiment must be logged with:

1. **Status:** Planned / Running / Completed (Succeeded|Failed) / Deferred
2. **Start date:** When the experiment began
3. **Compute cost:** GPU-hours consumed
4. **Primary result:** Primary metric value at evaluation point
5. **Key diagnostic:** The one diagnostic that explains the result
6. **Code branch:** Git branch where the experiment was run
7. **Checkpoint path:** Location of the best checkpoint
8. **Failed attempts:** If the experiment was iterated, what previous attempts failed

Recommended tracking format (to be maintained in a companion document or spreadsheet):

```
| Q# | Status | Start | GPU-h | Primary Result | Diagnostic | Branch | Ckpt |
|----|--------|-------|-------|----------------|------------|--------|------|
| Q1 | Planned | - | - | - | - | - | - |
| Q2 | Deferred | - | - | - | - | - | - |
| ... | ... | ... | ... | ... | ... | ... | ... |
```

The 3060 ablation experiments (25 epochs, ~80 GPU hours each) should be run in priority order. The 5060 Ti remains dedicated to the main run until Phase 2. The immediate inference-only experiments (Q8, Q42, Q48) can be completed today and logged immediately.

---

# Appendix G: Quantitative Budget for Remaining GPU Time

## Available GPU Resources

**RTX 5060 Ti 16GB** (main training GPU, 111-overview.md:241-254):
- Currently running main run (epoch 12 of 100, started ~July 2)
- Estimated completion: July 12-16
- Available for new experiments after main run completion
- Total remaining: ~200 GPU-days (July 16 through AAIML deadline)
- Effective training speed: ~0.6 batch/s at B=4 with NUM_WORKERS=0

**RTX 3060 12GB** (idle, 111-overview.md:249-254):
- Currently idle (crashed during ablation, restarted with B=4)
- Available immediately for ablation experiments
- Total remaining: ~30 GPU-days before AAIML deadline
- VRAM constraint: 12 GB limits B=2-4 depending on architecture complexity
- Effective training speed: ~0.4 batch/s at B=2

## Experiment Budget Allocation

Phase 0 (immediate inference, 0 GPU-days): Q8 (SWA), Q42 (Kalman), Q48 (hysteresis). Cost: essentially free. Priority: immediate.

Phase 1 (3060 ablations, ~10 GPU-days): 5 experiments at ~2 GPU-days each. Allocation:
- Q41 6D rotation: 2 days. Expected benefit: -0.8 to -1.6 deg MAE.
- Q21 CAGrad: 2 days. Expected benefit: +0.04-0.09 act F1.
- Q38 detection-aug act: 2 days. Expected benefit: +0.03-0.08 act F1.
- Q35 multi-scale train: 2 days. Expected benefit: +0.03-0.07 mAP50_pc.
- Q12 seq contrastive PSR: 2 days. Expected benefit: +0.10-0.25 PSR F1.

Phase 2 (5060 Ti after main run, ~30 GPU-days):
- Best Phase 1 combination as 100-epoch run: ~10 days
- Q31 Objects365 pretrain + fine-tune: ~5 days
- Remaining: distributed among Q1 (FCMAE), Q17 (FixMatch), Q46 (transition detection)
- Each experiment must first pass a 3060 15-epoch probe before committing to full 5060 Ti run

## Budget Constraint Summary

Total remaining GPU time before AAIML deadline (estimated late January 2027): ~230 GPU-days across both GPUs. At ~3 days per 25-epoch ablation and ~10 days per 100-epoch full run, we can execute approximately:
- 60 ablation experiments on the 3060 (30 GPU-days available)
- 20 full runs on the 5060 Ti (200 GPU-days available)

This is sufficient to execute all Phase 0-2 experiments plus leave margin for iterations and unexpected failures. The key decision is prioritization: each Phase 1 ablation must produce clear evidence before a Phase 2 full run is committed.

---

*End of document 125-50-deep-questions-sota.md. Cross-references: 111-overview.md (all project context), 112-training-metrics-deep-dive.md (gradient/loss data), 113-all-fixes-chronicle.md (fix history), 114-comparability-vs-4-papers.md (SOTA comparisons), 115-execution-plan-to-sota.md (timeline), 116-winning-aaiml-synthesis.md (paper positioning), 117-50-deep-questions-for-sota.md (prior questions), 118-opus-answers-111-117.md (verdicts on prior questions), 123-plan-to-compare-papers.md (comparison plan), 124-architecture-deep.md (architecture analysis).*

---

# Appendix H: Detailed Metric Definitions and Measurement Protocols

To ensure all 50 experiments produce comparable results, every metric must be measured using an identical protocol. This appendix defines each metric precisely, including edge cases and failure modes that must be handled identically across experiments.

## Detection Metrics

**det_mAP50 (COCO-24):** Mean Average Precision at IoU threshold 0.5 across all 24 ASD classes. Computed using the standard COCO evaluation protocol (averaging over classes, then over IoU thresholds 0.5 to 0.95 with step 0.05 for the full COCO metric, but only IoU=0.50 for the AP50 variant). All 24 classes are included in the average regardless of whether they have ground-truth instances in the validation subsample. Implementation: torchvision.detection.evaluate or pycocotools.

**det_mAP50_pc (present-class):** Same as det_mAP50 but computed only over classes that have at least one ground-truth instance in the current validation sample. The number of present classes (n_present) must be reported alongside. This metric was introduced to address the validation subsampling artifact where 9 of 24 channels have zero GT instances in the 250-batch subsample (111-overview.md:683-687). After D3 (full 38K validation), n_present should increase.

**Per-class AP:** Average Precision at IoU=0.50 for each of the 24 ASD channels individually. Reported as a 24-element vector. Critical for diagnosing which channels drive any overall mAP change.

**det_cls_mean:** Mean classification logit across all predictions. A drift toward negative values indicates the classifier is becoming more confident about "background" predictions (118 Section 1, 7.2). Tracked per-epoch as a leading indicator of OHEM+FocalLoss suppression.

**det_anchor_quality:** Mean IoU of matched anchors (positive anchors) to GT boxes. Healthy values are 0.85-0.90. Lower values indicate poor anchor matching (either anchor shape mismatch or regression head not learning).

## Activity Metrics

**act_macro_F1:** Macro-averaged F1 score across all 69 verb-grouped classes. Macro averaging means each class contributes equally regardless of its frequency. This is the primary activity metric because it is not biased toward common classes. Implementation: compute F1 per class, then average. Classes with zero predictions contribute F1=0.0 to the average (do not exclude them).

**act_Top1:** Per-frame top-1 accuracy: what fraction of frames have the correct class as the highest-probability prediction. For the per-frame MLP head (current), this is computed on individual frames. For temporal models (Q36, Q40), this is computed on clip-level predictions.

**act_Top5:** Per-frame top-5 accuracy: what fraction of frames have the correct class among the top 5 highest-probability predictions. This metric provides a more forgiving assessment of whether the model is in the right "action family" even when it cannot distinguish fine-grained actions.

**act_pred_distinct:** Number of distinct classes (out of 69) that are predicted at least once in the validation set. Low values (current: 35/69) indicate the model is collapsing predictions to a subset of classes. Increasing this number toward 69 is a positive leading indicator.

**act_clip:** Clip-level accuracy computed by majority voting over 16-frame windows (T4 metric, 117 Q42). Distinguish from act_frame (per-frame accuracy). These two numbers can diverge significantly on temporal data.

## Ego-Pose Metrics

**pose_fwd_MAE:** Mean Angular Error between predicted and ground-truth forward unit vectors (3D). Computed as: MAE = mean(arccos(clamp(dot(v_pred, v_gt), -1, 1))). The arccos is in degrees. Values below 5 deg approach the estimated HoloLens 2 sensor noise floor (116-winning-aaiml-synthesis.md:573).

**pose_up_MAE:** Same computation as forward MAE but for the up unit vector. Typically lower than forward MAE because up direction (gravity-relative) is more constrained by the IMU.

**pose_position_MAE:** Mean Absolute Error for the 3D position prediction (meters). Currently flagged "DO NOT USE FOR REPORTING" (evaluate.py:1918-1926) due to uncalibrated scaling. Q12 proposes removing this from the loss entirely.

**pose_gamma/beta_stats:** For the FiLM layer (Q13 analysis), the mean and standard deviation of gamma (scale) and beta (shift) parameters across all FiLM channels. Near-identity: gamma_mean ~1.0, beta_mean ~0.0, gamma/beta_std ~0.0.

## PSR Metrics

**PSR_POS:** Procedure Order Score. Edit-distance-based measure of how well the predicted component activation order matches the ground-truth canonical order. Range [0, 1]. Current: 0.968 beats SOTA 0.812 (STORM-PSR) and 0.797 (B3). The fill-forward paradigm inflates this metric (guarantees monotone ordering), hence the need for the canonical-order baseline (Q43).

**PSR_F1@3:** Per-component F1 score with +/-3 frame tolerance. A predicted transition is considered correct if it occurs within 3 frames of the ground-truth transition. Computed per component and macro-averaged. Current: 0.144 trails SOTA 0.901 (STORM-PSR) and 0.883 (B3). This is the primary PSR metric that must improve.

**PSR_edit:** Edit distance between the predicted and ground-truth state sequences. Lower is better. Current: 0.752.

**PSR_comp_acc:** Per-component accuracy (what fraction of frames have the correct presence/absence for each component). Current: 0.346.

**PSR_tau:** Mean per-component detection delay in seconds (frames / fps). Currently not measured (E2, 111-overview.md:186). B3 tau = 22.4s, STORM-PSR = 15.5s. Our fill-forward paradigm estimates 0.5-1.5s (10-15x faster) but this must be measured (Q17/Q44).

**PSR_confusion_matrix:** 11x11 per-component confusion matrix showing which components are most often confused. Critical for diagnosing the cascading error chain identified in 112.

## Combined Metric

**combined:** Weighted combination of task metrics used for model selection. Formula: combined = 0.33 * (mAP50 + mAP50_pc + 0.5*act_macro_F1 + 0.5*PSR_F1 + c). The exact formula is as defined in the project's config. During Phase A/B/C, the formula was different (118 Section 2, Anomaly 5). Current combined at epoch 11: 0.363.

**Note on combined metric evolution (118 Section 2, Anomaly 5):** Phase A/B/C combined values are not comparable to Phase D values due to multiple correctness fixes (F18, F22/F22b). All comparisons in this document use Phase D (stage_rf4) metrics as the baseline.

## Evaluation Pipeline Settings

All experiments must use the following evaluation settings for comparability:
- NMS IoU threshold: 0.5 (standard)
- Soft-NMS: disabled for baseline comparisons (Q1 tests this explicitly)
- EVAL_MAX_BATCHES: 0 for final evaluation, 250 for rapid iteration
- Confidence threshold: 0.05 (standard COCO evaluation)
- Max detections per image: 100 (standard)
- Input size: 224x224 (except Q35 multi-scale which varies this)

## Gradient and Internal Diagnostics

For ablation experiments, the following internal diagnostics must be logged at every 5 epochs:
- Per-task gradient composition (det %, pose %, act %, psr %)
- Per-task loss values (raw, unweighted)
- Kendall log_var values (if Kendall weighting is active)
- Per-class AP for detection (24 values)
- Per-component F1 for PSR (11 values)
- Gradient cosine similarities for all task pairs (6 pairs)
- Learning rate (current value from scheduler)
- EMA model metrics (vs raw model metrics)

These diagnostics are essential for understanding why an experiment succeeded or failed, and for the negative result analysis template (Appendix E).

---

# Appendix I: The 7 Claims That Can Beat SOTA

Each claim maps to multiple questions in this document. The probability of each claim succeeding is estimated based on the cumulative probability of the underlying questions succeeding.

## Claim 1: Detection Beats YOLOv8m on Present-Class Metric (mAP50_pc)

**Supporting questions:** Q1 FCMAE, Q3 CST, Q4 SimOTA, Q5 NAS-FPN, Q6 curriculum, Q11 TAFL, Q16 tubelet, Q17 FixMatch, Q18 counterfactual, Q19 adaptive curriculum, Q31 Objects365, Q32 DINOv2, Q34 distillation, Q35 multi-scale.

**Path to success:** The highest-probability path combines Q31 (Objects365 pretrain, +0.06-0.12 mAP50) + Q35 (multi-scale train, +0.03-0.07 mAP50_pc) + Q34 (distillation, +0.03-0.08 mAP50). This path has a 40-60% estimated probability of achieving mAP50 > 0.45 and mAP50_pc > 0.60.

**Probability estimate:** 35-50% chance of beating YOLOv8m on present-class metric (requires mAP50_pc > 0.838 to truly beat -- very unlikely). More realistic: 70-80% chance of closing detection gap to < 0.25 (mAP50 > 0.55).

## Claim 2: Activity Beats MViTv2 on Verb-Grouped Metric

**Supporting questions:** Q20 optical flow, Q36 hierarchical TCN, Q37 verb-noun hierarchy, Q38 detection-aug act, Q39 SMOTE, Q40 VideoMAE dual-backbone.

**Path to success:** The highest-probability path combines Q37 (verb-noun hierarchy, +0.04-0.09 macro-F1) + Q38 (detection-aug, +0.03-0.08 macro-F1) + Q39 (SMOTE, +0.02-0.06 macro-F1). This path has a 30-50% estimated probability of achieving macro-F1 > 0.20.

**Probability estimate:** 20-30% chance of matching remapped MViTv2 (0.25-0.35 macro-F1). 50-60% chance of achieving macro-F1 > 0.18.

## Claim 3: Ego-Pose Beats All Baselines Toward HL2 Floor

**Supporting questions:** Q13 uncertainty geodesic, Q41 6D rotation, Q42 Kalman filter, Q43 multi-scale pose, Q44 IKEA pose aug, Q45 MC dropout.

**Path to success:** The highest-probability path combines Q41 (6D rotation, -0.8 to -1.6 deg) + Q42 (Kalman, -0.3 to -0.8 deg) + Q13 (uncertainty geodesic, -1.0 to -1.6 deg). This path has a 50-70% estimated probability of achieving forward MAE < 6.5 deg.

**Probability estimate:** 50-70% chance of forward MAE < 7.0 deg. 25-40% chance of forward MAE < 6.0 deg (approaching HL2 noise floor).

## Claim 4: PSR POS Beats SOTA with Paradigm Disclosure

**Supporting questions:** Q43 canonical-order baseline (diagnostic only), Q46 transition detection, Q47 temporal cross-attention.

**Path to success:** This claim is already supported (POS 0.968 vs SOTA 0.812). The canonical-order baseline (Q43) determines the strength of the claim. If POS drops to 0.88-0.92 under transition detection (Q46), it still beats SOTA 0.812.

**Probability estimate:** 95%+ chance of beating SOTA regardless of experimental outcomes (current POS=0.968 has 0.15 margin over SOTA 0.812).

## Claim 5: PSR F1 Beats SOTA Through Temporal Modeling

**Supporting questions:** Q12 seq contrastive, Q14 order-reg, Q46 transition detection, Q47 temporal cross-attention, Q48 hysteresis, Q49 quality-adaptive, Q50 multi-decoder ensemble.

**Path to success:** The highest-probability path combines Q46 (transition detection, +0.20-0.40 F1) + Q48 (hysteresis, +0.06-0.14 F1) + Q12 (contrastive, +0.10-0.25 F1). This path has a 20-35% estimated probability of achieving PSR F1 > 0.50.

**Probability estimate:** 30-45% chance of achieving PSR F1 > 0.35. 15-25% chance of achieving PSR F1 > 0.50 (competitive with B3's 0.883). Reaching STORM-PSR's 0.901 requires additional breakthroughs.

## Claim 6: Single-GPU Multi-Task Efficiency Thesis

**Supporting questions:** Q7 SAM (flatter minima with 2x train cost), Q9 LLRD (efficient fine-tuning), Q33 ConvNeXt-Nano (smaller model), Q21-25 (multi-task balancing).

**Path to success:** This claim is thesis-level (115-execution-plan-to-sota.md:88-92). It requires: (a) performance comparable to dedicated-single-task models, (b) 4x task efficiency (one model doing 4 tasks), (c) single GPU inference. The ablations (A1-redo, A2-A4 from 118) provide the evidence.

**Probability estimate:** 60-80% chance of a defensible efficiency claim after corrected ablations. The thesis is primarily narrative (4 tasks on one GPU) rather than metric-driven.

## Claim 7: Novel Method Contributions (AAIML Accept Criteria)

**Supporting questions:** Q11 TAFL, Q12 seq contrastive PSR, Q15 multi-task NT-Xent, Q30 universal assembly rep, Q37 verb-noun hierarchy, Q50 multi-decoder ensemble.

**Path to success:** Any 2-3 of these novel methods succeeding provides the methodological novelty required for AAIML acceptance. The most likely methods to succeed: Q11 TAFL (binary-code-aware detection loss) and Q37 verb-noun hierarchy (compositional activity recognition).

**Probability estimate:** 40-60% chance of 2+ novel methods succeeding. AAIML acceptance requires at least 2 novel contributions plus strong metrics.

---

## Overall SOTA Assessment

The project is currently strong on PSR POS (beat SOTA by 19%) and ego-pose (first baseline). It is weak on detection (62% gap to YOLOv8m), activity (55-69% gap to remapped MViTv2), and PSR F1 (84% gap to STORM-PSR). The 50 questions in this document are designed to close these gaps through targeted improvements.

The most efficient path to SOTA is: (1) immediate inference-only wins (Kalman, hysteresis, SWA), (2) 3060 ablations of the most promising approaches (CAGrad, 6D pose, multi-scale training, contrastive PSR), (3) 5060 Ti integrated run combining the winners. This path maximizes the probability of beating SOTA on at least 4 of 5 metrics while maintaining the already-winning POS and first-baseline ego-pose claims.

---

*Total: 50 questions across 10 categories (5 each), plus 9 appendices. Each question follows the standard format with Context, Question, Why This Matters, Constraints, Hypothesis, and Validation sections. Every question cites specific project documents (111-118) with line numbers, specifies quantitative metric estimates, and defines concrete validation experiments.*

*This document replaces 117-50-deep-questions-for-sota.md as the active question set. Older questions from 117 are superseded by the deeper, more specific questions in this document.*

---

# Appendix J: Mathematical Formulations of Key Loss Functions

This appendix provides the complete mathematical formulation for all novel loss functions proposed in this document. Each formulation includes the forward computation, gradient properties, and numerical stability considerations.

## J.1 Transition-Aware Focal Loss (Q11)

Let C be the 24x24 ASD class transition cost matrix, where C[i][j] = Hamming(binary[i], binary[j]) * violation_penalty(i,j). The violation penalty is 2.0 if the transition violates assembly order (component appears before its prerequisite), else 1.0.

Standard FocalLoss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), where p_t is the predicted probability for the true class t.

Transition-Aware Focal Loss:
TAFL(p) = sum_{c=1}^{24} C[t][c] * FL(p_c)

where p_c is the predicted probability for class c, and t is the true class. This penalizes each prediction proportionally to its transition cost: predicting a class that is far in binary code space (Hamming distance > 3) or violates assembly order (violation_penalty=2.0) receives higher loss even if it is not the top prediction.

Gradient: d(TAFL)/d(p_c) = C[t][c] * d(FL)/d(p_c). For c=t, C[t][t] = 0 (the true class incurs no transition penalty). This means the gradient from TAFL pushes probability mass away from high-cost false classes, rather than toward the true class. The standard FL gradient toward the true class is preserved by the c=t term where C[t][t]=0.

Implementation note: C should be normalized so that mean(C[t][j] for j != t) = 1.0 for all t, ensuring the overall loss magnitude is comparable to standard FocalLoss.

## J.2 Sequence-Level Contrastive PSR Loss (Q12)

Let z_t = backbone_features(frame_t) be the feature vector at time t. Define positive pairs as (z_t, z_{t+1}) when the assembly state is the same at both t and t+1. Define negative pairs as (z_t, z_s) where frames t and s are from different assemblies or from assembly positions where the state differs by >2 components.

The InfoNCE loss for a positive pair (z_i, z_j) with temperature tau:

L_contrastive = -log( exp(sim(z_i, z_j)/tau) / (exp(sim(z_i, z_j)/tau) + sum_{k in N(i)} exp(sim(z_i, z_k)/tau)) )

where sim(a,b) = dot(a,b) / (||a|| * ||b||) is cosine similarity, and N(i) is the set of negative samples for anchor z_i.

In practice, for each mini-batch of B sequences of length T:
1. Encode all B*T frames through backbone to get features z_{b,t}
2. For each anchor frame (b,t), positive = (b,t+1) if state is same, else skip
3. Negative samples: all (b',t') where b' != b (different assembly) or state differs by >2 components
4. Compute InfoNCE loss for all positive pairs
5. Total loss = L_BCE(PSR predictions, PSR labels) + lambda * L_contrastive

Temperature tau to sweep {0.07, 0.1, 0.5}. Contrastive weight lambda to sweep {0.1, 0.3, 0.5}.

## J.3 Uncertainty-Aware Geodesic Loss (Q13)

Let v_pred and v_gt be 3D unit vectors. The geodesic (angular) error:
theta = arccos(clamp(dot(v_pred, v_gt), -1, 1))

Standard geodesic loss: L_geo = theta

Uncertainty-aware geodesic loss with learned per-sample variance sigma (one scalar per prediction, learned as log(sigma^2) for numerical stability):

L_uncertainty_geo = theta / sigma^2 + 0.5 * log(sigma^2)

The first term down-weights high-variance predictions; the second term prevents sigma from growing unbounded (regularization). The optimal sigma* for each sample equals the angular error: sigma* = theta-empirical.

For 3D vectors, the output dimension is 7: (log_sigma_fwd, log_sigma_up, v_fwd_xyz, v_up_xyz). The total loss is the sum of fwd and up uncertainty-weighted geodesic losses.

Gradient: dL/dv_pred = (dtheta/dv_pred) / sigma^2. The 1/sigma^2 term rescales the gradient, making high-uncertainty predictions update more slowly.

## J.4 PSR Component-Order Regularization (Q14)

Let A be the 11-component canonical assembly order (list of component indices in order of first activation). Let tau_empirical be the median detection frame for each component (measured on the current batch of predictions).

The order-regularization loss:

L_order = sum_{i=1}^{10} sum_{j=i+1}^{11} w_i * w_j * max(0, tau_empirical[A[i]] - tau_empirical[A[j]])^2

where w_i = 1 / prevalence(component A[i]) is the inverse-prevalence weight. This penalizes any pair where a later-in-order component (A[j]) is detected before an earlier-in-order component (A[i]), weighted by the inverse prevalence (rare components get higher weight because their detection order matters more for the cascade).

Components with zero detections: tau_empirical = max(tau_observed) + penalty (they are treated as detected after all detected components, incurring full regularization loss).

Numerical stability: clamp the gradient to [-10, 10] to prevent rare-component large weights from producing exploding gradients.

## J.5 Multi-Task NT-Xent Loss (Q15)

For each input image x, generate two augmentations x' and x''. Pass both through backbone to get features f' and f''.

For a batch of N samples, this produces 2N feature vectors. The multi-task NT-Xent loss:

L_NT_Xent = -(1/2N) * sum_{i=1}^{2N} log( exp(sim(f_i, f_{i+N})/tau) / sum_{j != i} exp(sim(f_i, f_j)/tau) )

where sim(a,b) = dot(a,b) / (||a|| * ||b||), and the positive pair (f_i, f_{i+N}) are the two augmentations of the same original image.

The total loss: L_total = L_det + L_pose + L_act + L_psr + lambda * L_NT_Xent

where lambda controls the contrastive regularization strength. The NT-Xent loss encourages the backbone to produce discriminative features that are invariant to augmentation, reducing task-specific feature specialization.

## J.6 GradVac Gradient Surgery (Q25)

Let g_i and g_j be the gradient vectors for tasks i and j (backbone portion only). The cosine similarity:
cos_ij = dot(g_i, g_j) / (||g_i|| * ||g_j||)

GradVac maintains an exponential moving average of historical cosine similarities:
EMA_ij = beta * EMA_ij_prev + (1 - beta) * cos_ij

If cos_ij < target_sim (the target minimum similarity), GradVac projects g_j to increase its similarity with g_i:

g_j_proj = g_j - (1 - target_sim / max(cos_ij, epsilon)) * proj_{g_i}(g_j)

where proj_{g_i}(g_j) = dot(g_j, g_i) / (||g_i||^2) * g_i is the projection of g_j onto g_i.

The final combined gradient: g_combined = g_1 + sum_{i=2}^{4} g_i_proj

This ensures all task gradients maintain at least target_sim cosine similarity with task 1 (the primary task). In practice, the gradient surgery is applied to all task pairs, not just task 1, using a similar procedure.

Implementation for 4 tasks:
1. Compute all 6 pairwise cosine similarities
2. For each pair with cos < target_sim, project the smaller-norm gradient
3. Average the projected gradients
4. Combine with original weighting if needed

## J.7 CAGrad for 4-Task Balancing (Q21)

Let g_1, g_2, g_3, g_4 be the per-task gradients (backbone portion). Let G be the 4xd matrix stacking these gradients (d is the number of backbone parameters).

CAGrad solves at each step:

min_{g} ||g - g_0||^2  subject to  min_i dot(g_i, g) >= 0

where g_0 is the average gradient: g_0 = (1/4) * sum g_i.

The constraint ensures g has non-negative dot product with all task gradients (improves all tasks). The objective ensures g is close to g_0 (doesn't deviate too far from the average direction).

The Lagrangian solution (from Theorem 1 of Liu et al., 2021):

g = g_0 + max(0, (1 - alpha) * g_0_omega )  where g_0_omega is the projection of negative cosines

In practice, CAGrad is implemented as an inner optimization (solved via bisection) that takes <1ms per step:

```
def cagrad(g_list, alpha=0.5):
    g0 = sum(g_list) / len(g_list)
    # Compute the worst-case direction
    g = g0.clone()
    for _ in range(10):  # Inner optimization (few iterations suffice)
        cos = [dot(g, g_i) / (norm(g) * norm(g_i)) for g_i in g_list]
        g = g0 + alpha * sum(max(0, -c) * g_i for c, g_i in zip(cos, g_list))
    return g
```

---

# Appendix K: Ablation Isolation Protocol

To ensure each experiment measures the intended effect without confounding from other changes, this protocol defines how experiments must be isolated. Violating this protocol invalidates the experiment and requires a re-run.

## K.1 Single-Variable Change Rule

Each experiment must change exactly one variable from the baseline configuration. All other hyperparameters, architecture components, data processing steps, and evaluation settings must be identical.

Allowed changes:
- One config parameter change (e.g., learning rate, loss weight, architecture flag)
- One code module replacement (e.g., replace FPN with CST neck, all other modules unchanged)
- One data processing change (e.g., add augmentation, change sampling strategy)

Disallowed:
- Changing multiple config parameters simultaneously
- Combining architecture and data changes in one experiment (run separately)
- Changing the evaluation pipeline when measuring training effects (and vice versa)

## K.2 Baseline Synchronization

All experiments in a category must use the same baseline. If the baseline is updated (e.g., bug fix, new best checkpoint), all in-progress experiments must be restarted from the updated baseline.

**Exception:** Immediate inference-only experiments (Phase 0) use the current epoch-11 checkpoint and do not need to be re-run if the baseline improves (they only add improvements).

## K.3 Seed Control

All 3060 ablation experiments use SEED=42 (matching the current baseline). Multi-seed runs (Q15 in 117, not in this document) use additional seeds 7 and 123 but are separate experiments.

If an ablation produces a result close to the baseline (within 0.5 standard deviations estimated from prior variance), run a second seed (SEED=7) to distinguish statistical noise from a real effect.

## K.4 Checkpoint Management

Each experiment must:
1. Save checkpoints every 5 epochs (minimum)
2. Log best checkpoint based on validation combined metric
3. Use a clean checkpoint directory (not shared with other experiments)
4. Store experiment config alongside checkpoints (copy of all config values at start)

Checkpoint lifecycle:
- Raw checkpoints: kept for all epochs (required for SWA post-hoc averaging)
- Best checkpoint: updated whenever validation combined improves
- Resume checkpoint: updated every 5 epochs (for experiment continuation if interrupted)

## K.5 Result Reporting Standard

Each experiment result must be reported as:
- Primary metric value at evaluation point (e.g., det_mAP50 at epoch 25)
- Standard deviation across evaluation batches (computed via bootstrap on validation set)
- Comparison to baseline at the same epoch (delta + significance)
- Confidence interval: 95% CI computed via 1000 bootstrap resamples

This reporting standard enables proper statistical comparison between experiments and prevents over-interpretation of noise.

---

*End of document 125-50-deep-questions-sota.md. Total: 50 questions across 10 categories (5 each), 11 appendices (A-K), comprehensive cross-referencing, validation protocols, mathematical formulations, and execution prioritization.*

*All questions cite specific project documents (111-overview.md, 112-training-metrics-deep-dive.md, 113-all-fixes-chronicle.md, 114-comparability-vs-4-papers.md, 115-execution-plan-to-sota.md, 116-winning-aaiml-synthesis.md, 117-50-deep-questions-for-sota.md, 118-opus-answers-111-117.md) with line numbers where available.*

*Execution priority: Phase 0 (inference-only, today) > Phase 1 (3060 ablations, this week) > Phase 2 (5060 Ti full runs, post-July 16). Each phase gates on: Phase 0 results must be incorporated before Phase 1 planning; Phase 1 results determine which experiments advance to Phase 2.*
