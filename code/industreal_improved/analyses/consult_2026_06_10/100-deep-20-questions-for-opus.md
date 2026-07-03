# 100 -- Twenty Deep Technical Questions for Opus

> **Based on:** RF4 training data (RTX 5060 Ti / Blackwell / CUDA 13.0 / driver 595.71.05)
> **Source file:** `/home/newadmin/swarm-bot/.superpowers/plans/opus-consult-rf4/97-current-status-deep-analysis.md`
> **Code base:** `/home/newadmin/swarm-bot/.claude/teams/industreal-deep-audit/agent1-training-loop.md` through `agent20-integration.md`
> **Training code:** `/media/newadmin/master/POPW/working/code/industreal_improved/src/training/train.py` (~4,431 lines)
> **Model code:** `/media/newadmin/master/POPW/working/code/industreal_improved/src/models/model.py` (~2,100 lines)
> **Loss functions:** `/media/newadmin/master/POPW/working/code/industreal_improved/src/training/losses.py` (~1,620 lines)
> **Configuration:** `/media/newadmin/master/POPW/working/code/industreal_improved/src/config.py` (~1,380 lines)
> **Evaluation:** `/media/newadmin/master/POPW/working/code/industreal_improved/src/training/evaluate.py` (~4,297 lines)
> **Date:** 2026-07-03

---

## Section 1: GPU Stability Crisis (Q1--Q4)

---

### Q1: Blackwell + CUDA 13.0 Driver Timeout Root Cause

**Context:** The RTX 5060 Ti (Blackwell architecture, 16 GB) is paired with CUDA 13.0 and driver 595.71.05. Every training run dies between epoch 3--5 with `cudaErrorLaunchTimeout`. The applied mitigations are: `CUDNN_BENCHMARK=False` (config.py `CUDNN_BENCHMARK`), `NVIDIA_TF32_OVERRIDE=0`, `TORCH_CUDNN_V8_API_DISABLED=1`, and `ALLOW_TF32=False` (enforced at `torch.set_float32_matmul_precision('highest')`). These reduce but do not eliminate the timeout. The crash always occurs during training (forward+backward on a seq batch or activity-heavy batch), never during idle periods. The GPU is a single consumer card (not TCC mode). `CUDA_LAUNCH_BLOCKING` was explicitly avoided because it would slow training by 30--50% (agent8-memory-performance.md line 529).

**The question:** Is the `cudaErrorLaunchTimeout` on Blackwell + CUDA 13.0 a known driver bug with the 595.xx series, specifically affecting kernel execution time estimation for persistent kernel loops? The 3060 (Ampere, driver 550 series) has never exhibited this. Rolling back to the 550 driver series (which Ampere uses) or to CUDA 12.x would be the conservative fix -- but is there a Blackwell-specific workaround (e.g., `CUDA_DEVICE_MAX_CONNECTIONS=1`, `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)`, or `NVIDIA_CUDA_LAUNCH_OPTIMIZER=0`) that preserves CUDA 13.0 compatibility? Reference: train.py lines 1581--1625 (gradient clipping and scaler step are the specific code paths where the watchdog fires).

---

### Q2: Activity Loss Surge Causing Watchdog at Epoch 5 Batch ~100

**Context:** The watchdog consistently fires at epoch 5, batch ~100. In the epoch 4 loss tables, activity loss rises from 1.76 to 5.04 across the epoch (status file epoch 4 batch tables: act=3.58 at batch 168, act=5.04 at batch 4835). The activity ramp (ACT_RAMP_EPOCHS=3, config.py `ACT_RAMP_EPOCHS`) completes at epoch 3, so by epoch 5 the activity head is at full strength with the OneCycleLR still warming up (peak LR at pct_start=0.1, approximately epoch 12). The F10 fix raised `ACTIVITY_HEAD_GRAD_CLIP` from 1.0 to 5.0 (config.py `ACTIVITY_HEAD_GRAD_CLIP`), but sequential gradient scaling (the `scaler.scale(loss).backward()` at train.py line ~1150, then `scaler.unscale_()` at line ~1153, then `clip_grad_norm_` at line ~1178 on the seq path, or lines 1587--1625 on the non-seq path) means the activity head's gradient surge could create a very large combined gradient norm that triggers a long kernel.

**The question:** Is the causal chain: (a) activity loss spikes (5.0+) create large per-sample gradients, (b) the gradient scaler amplifies them further, (c) `clip_grad_norm_(model.parameters(), max_norm=1.0)` at train.py line 1622 spends abnormal wall time computing the L2 norm over ~30M+ parameters because some gradient entries are very large (documented in agent6-gradient-flow.md lines 65--69: the gradient vector has "well over 10^7 elements" and a cap of 1.0 means gradients are suppressed to vanishing levels when even 2--3 heads produce moderate gradients), (d) the cumulative step time exceeds the 1,800-second watchdog? If this chain is correct, would applying `torch.nn.utils.clip_grad_value_` (value-based, not norm-based) as a pre-filter BEFORE the norm computation avoid the timeout? Or does the Blackwell watchdog measure actual kernel execution on GPU (not CPU wall time), making the gradient-clipping hypothesis incorrect?

---

### Q3: X Server Interference and Multi-GPU Workaround

**Context:** There are two GPUs physically present: RTX 5060 Ti (16 GB, Blackwell) for compute, and RTX 3060 (12 GB, Ampere) driving the display (Xorg/GDM). The 3060 is completely invisible to CUDA (CUDA_VISIBLE_DEVICES is not set, but `torch.cuda.device_count()` returns 1). The display manager runs on the 3060 via PCI bus isolation (primary GPU is the 3060). The 5060 Ti has no display output. The `cudaErrorLaunchTimeout` is a GPU-wide event that resets the entire GPU, which could affect all processes on that GPU but NOT the 3060 display GPU.

**The question:** Could the X server on the 3060 be causing system-level PCIe TLP (Transaction Layer Packet) pressure that triggers the Blackwell GPU's heartbeat timeout? Specifically, if both GPUs share a PCIe root complex (typical consumer motherboard), display operations on the 3060 could delay PCIe completion for kernel launches on the 5060 Ti. A diagnostic would be: stop gdm, run headless, and check if the timeout disappears. If yes, the fix is permanent (run headless). If no, the issue is truly Blackwell/driver. Is there precedent for Blackwell consumer GPUs needing the display manager to be stopped to avoid TLP-induced launch timeouts, or is this purely a driver issue?

---

### Q4: BF16/AMP as a Watchdog Mitigation

**Context:** Fix F6 added BF16 autocast support (`AMP_DTYPE='bf16'`, config.py `AMP_DTYPE`) but it has never been tested because `MIXED_PRECISION=False` (config.py `MIXED_PRECISION`) is the current default for stability. The Blackwell architecture natively supports BF16 compute (faster than FP32 matrix multiplies). The kernel timeout measures GPU kernel execution time -- if BF16 halves the execution time of the 10--20 largest matrix multiplies per step, the cumulative step time could drop below the 1,800-second watchdog threshold (current step time is ~1.7--2.0 seconds at batch=4, documented in status file section 6 GPU state).

**The question:** If we enable BF16 AMP, what is the expected reduction in kernel execution time per step for this specific model (ConvNeXt-Base backbone + MViTv2 temporal encoder + 5 task heads)? The ConvNeXt-Base has ~88M parameters with most compute in `nn.Conv2d` layers which are memory-bound at batch=4. MViTv2's temporal encoder uses `nn.Conv3d` (bandwidth-bound). The task heads are linear/dense (compute-bound). For Blackwell, which layers benefit most from BF16 (matrix multiply heavy) and which are bottlenecked by memory bandwidth (no BF16 benefit)? Could the specific layer where the timeout occurs be identified by examining `torch.cuda.CUDAPluggableAllocator` traces or Nsight Systems profiling?

---

## Section 2: Detection and Activity (Q5--Q8)

---

### Q5: Detection mAP50_pc Trajectory After F1 Fix

**Context:** At epoch 2 validation, `det_mAP50_pc=0.133` with 15 of 24 classes detected (status file section 4.1). In RF2 (which ran with the F1 bug active -- backbone grad wipe destroyed ~80% of backbone gradient according to the status file F1 description: "Seq-batch backbone grad wipe removed (was destroying ~80% backbone signal)"), `det_mAP50_pc` reached 0.31 by epoch 21. The model uses Pose-Derived Detection (PDD) via skeleton keypoints (`PoseDerivedDetection` class at model.py line 768), not a learned detection head. The PDD method is purely geometric: `worker_box_margin=10.0` pixels, `bottle_box_radius=60.0` pixels (model.py lines 797--798). The "mAP" metric measures skeleton keypoint accuracy projected into bounding boxes, NOT a learned detection function.

**The question:** With the F1 fix, the backbone now receives gradient from ALL heads on ALL batches (~5x more gradient signal per epoch). But PDD is a geometric function of keypoints -- it does NOT learn. So the detection "mAP" is entirely dependent on the pose head's keypoint accuracy. Since the pose head converged by epoch 2 (pose loss 0.03--0.15), having more backbone gradient does NOT improve PDD output. Is `det_mAP50_pc` already at its maximum given the current pose head accuracy, and should we expect it to plateau at 0.15--0.20 rather than climb to 0.30+ as in RF2? What is the theoretical maximum PDD mAP given a pose forward angular MAE of ~11 degrees and positional error of ~65 mm?

---

### Q6: Activity Loss Rising as Ramp Completes -- Fighting or Learning?

**Context:** Activity loss is rising from ~0.9 to ~5.0 as the activity ramp completes (ACT_RAMP_EPOCHS=3). The status file epoch-2 loss table shows act=2.28 at batch 750 rising to act=3.17 at batch 5417. By epoch 4, act ranges from 1.91 to 5.04. The current `lv_act` (Kendall log-variance for activity) has risen from 0 to 0.114 over 5,700 steps (status file section 5). At lv_act=0.114, the Kendall precision is `exp(-lv_act)` = ~0.89 (close to 1.0, meaning activity receives near-full weight). The consulting advice said to keep `ACTIVITY_LOSS_WEIGHT=0.8` and not increase it.

**The question:** A rising loss during ramp completion could mean two things: (a) the activity head is actively separating features (the cosine probe added at F12, `grad_cosine_probe.py` for offline task conflict analysis, would show increasing task conflict angles between activity and other heads) OR (b) the activity head is in a degenerate competition with other heads for shared feature space (backbone C5 via `c5_mod` at model.py line 2040, FPN P4 via `pyramid['p4']` at line 2041). An audit at agent12-activity.md lines 158--169 confirmed that these feature paths have NO stop_grad: "c5_mod and p4 do propagate gradients back to backbone and FPN parameters." If (b) is happening, the activity head is actively corrupting shared features. What value of `lv_act` would indicate that activity's precision is causing harm to other heads? Since `lv_act` crossing 0.5 would give precision = exp(-0.5) = 0.606 (reducing activity weight by 40%), would we expect the loss to begin falling at that point as the backbone rebalances?

---

### Q7: FOCAL_ALPHA=0.50 With gamma_pos=0 -- Theoretical Analysis

**Context:** Fix F8 changed `FOCAL_ALPHA` from 0.25 to 0.50 (config.py `FOCAL_ALPHA`, applied to activity loss). The focal loss implementation (losses.py `FocalLoss` class, line 102) uses `gamma=2.0` as default with a custom `gamma_pos=0` modification meaning the positive gradient is NOT down-weighted by focal. The effective per-sample gradient weighting is: **positives**: `alpha * CE` = 0.50 * CE. **Negatives**: `alpha * p^gamma * CE` = 0.50 * p^2 * CE. For a typical early-training negative with sigmoid output p=0.03: contribution = 0.50 * 0.0009 * CE = 0.00045 * CE. For a borderline negative with p=0.5: contribution = 0.50 * 0.25 * CE = 0.125 * CE.

**The question:** With `gamma_pos=0`, positives contribute 0.50*CE regardless of their confidence. Negatives contribute 0.50*p^2*CE, which at typical early sigmoid values (p~0.03) gives 0.00045*CE -- essentially zero. This creates a gradient landscape where positives dominate absolutely at the start. But as the classifier becomes confident (p_pos ~ 0.9), negatives become essentially silent (p_neg ~ 0.1, contribution = 0.50 * 0.01 * CE = 0.005*CE). Does this asymmetric weighting create a "hard negative starvation" problem where the classifier learns to identify positives but never learns to suppress hard negatives (p ~ 0.3--0.7)? The original RetinaNet paper (Lin et al., ICCV 2017) uses alpha=0.25 with gamma=2.0 and NO gamma_pos=0, giving both positive AND negative down-weighting. Is the `gamma_pos=0` modification mathematically unsound for a 69-class activity classifier with severe class imbalance?

---

### Q8: Activity Output Classes -- 69 Hybrid Groups

**Context:** The activity head outputs 69 logits (status file section 4.1: "Activity: pred_distinct=5/69 classes, entropy=1.270 nats"). The config file comment guesses "~41--47 groups" but the actual parser produces 69. The 69 output classes come from hybrid grouping of raw action IDs (numerical IDs 0--74, with ID 37 permanently absent per agent12-activity.md line 23: "ID 37 is absent from the dataset (permanent cold channel)"). The validation log shows `pred_distinct=5/69` at epoch 2, meaning only 5 of 69 possible groups are being predicted. Activity entropy is 1.270 nats (theoretical maximum for uniform 69-class = ln(69) = 4.234 nats).

**The question:** Is the hybrid group count of 69 correct, or does it reflect an over-splitting of the action space? Each group should have at least ~100 training examples for the classifier to learn meaningful boundaries. With the full IndustReal dataset of ~25,000 training frames and 69 classes, the average is ~362 frames per class. But the actual distribution is power-law (some actions appear thousands of times, others dozens). Should some low-frequency groups be merged back into the parent action class, reducing the effective output dimension to 30--40? What is the theoretical maximum number of linearly separable action groups for a 768-dim feature space (backbone C5 output) with 25K training examples?

---

## Section 3: PSR and Pose (Q9--Q12)

---

### Q9: PSR Convergence Rate at seq_every=4

**Context:** Fix F7 changed `PSR_SEQ_EVERY_N_BATCHES` from 2 to 4 (config.py `PSR_SEQ_EVERY_N_BATCHES`). Only 1 in 4 batches processes PSR. On non-seq batches, PSR loss is structurally zero (the `MultiTaskLoss` at losses.py line ~1317 skips PSR when `USE_PSR_TRANSITION=False` and dim==2). Additionally, `detach_psr_fpn=True` (config.py `DETACH_PSR_FPN`) isolates PSR from the backbone entirely -- PSR's `p3_t.detach()`, `p4_t.detach()`, `p5_t.detach()` at model.py lines 1957--1960 stop gradient flow to FPN features. On seq batches, train.py lines 1121--1131 zero the backbone and FPN gradients after backward, giving PSR ONLY its own head parameters and the transformer.

**The question:** The PSR head (~1--2M parameters including the causal transformer: 3 layers, 4 heads, d_model=256, FFN=1024 per agent10-psr.md line 14) receives gradient updates on only ~25% of batches, and even then only its head-internal parameters change. At this rate, with ~6,580 batches per epoch and ~1,645 PSR updates per epoch, how many epochs are needed for convergence of a randomly initialized transformer? The `PSR_WEIGHT=10.0` (agent10-psr.md line 25) multiplies PSR loss before Kendall, and `PSR_SEQ_LOSS_SCALE=1.5` (line 26) further amplifies seq batch loss. But `PSR_LOSS_CAP=20.0` (line 27) caps the total. With 15x amplification before Kendall and a cap at 20.0, is the effective PSR gradient magnitude in the same range as other heads, or is it negligible? When should `detach_psr_fpn` be flipped to False (presumably RF6+ per status file section 7)?

---

### Q10: PSR Per-Component Metrics with Fill-Forward Labels

**Context:** PSR predicts 11 per-component binary states plus confidence (12 outputs total, per model.py `STORMPSR` architecture). The labels are fill-forward: after a component transitions from 0 to 1, it stays 1 for all subsequent frames. Component prevalence ranges from 0.19 to 1.0 (component 0, base plate, has prevalence ~1.0 because it appears in every frame after initial placement). The evaluation metric `psr_f1_at_t` uses monotonic decoding via `MonotonicDecoder` (evaluate.py lines 302--365) which greedily matches 0->1 transitions between prediction and ground truth with +/-3 frame tolerance.

**The question:** With fill-forward labels, a model that predicts all-ones for all components achieves ~95% per-frame accuracy (because most frames have all components placed), but `psr_f1_at_t = 0.0` because it detects ZERO transitions (agent10-psr.md lines 184--190 document this exact failure mode). Is per-component binary accuracy a meaningful metric when the label distribution is 80--100% ones? Should the evaluation compute metrics only on transition frames (frames where at least one component transitions 0->1), giving a denominator of ~50--200 frames per video instead of ~3,000? What is the expected `psr_f1_at_t` for a random transition predictor given the fill-forward structure?

---

### Q11: lv_pose = -1.000 and HP_PREC_CAP

**Context:** `lv_pose` has been exactly -1.000 since epoch 2 and never changes (status file section 5: stepped at 101, 301, 2701, 3701, 4701, 5701 -- always -1.000). The status file interpretation says "HP_PREC_CAP active (pose capped at detection precision)." The HP_PREC_CAP mechanism (implemented in losses.py as part of `MultiTaskLoss`) enforces that pose precision never exceeds detection precision. The mechanism is implemented as a minimum-bound constraint on the log-variance: `lv_pose >= lv_det` after each Kendall update.

**The question:** The value `lv_pose = -1.000` means the Kendall precision for pose is `exp(-(-1.0)) = e^1 = 2.718`, which is HIGH precision (pose loss gets multiplied by ~2.7). But `lv_det` started at 0.004 and has risen to 0.075, meaning detection precision = `exp(-0.075) = 0.928`. If the cap is `lv_pose >= lv_det`, the constraint is `-1.000 >= 0.075` which is FALSE -- the cap should trigger and raise lv_pose to at least 0.075. But lv_pose never changes. Is the HP_PREC_CAP implemented as a lower bound on lv_pose (preventing lv_pose from going below lv_det, i.e., pose from having MORE precision than detection) -- in which case the current state VIOLATES the cap? Or is the comparison inverted: `lv_pose >= lv_det` is ALREADY SATISFIED when lv_pose = -1.000 and lv_det = 0.075 because -1.000 >= 0.075 evaluates to TRUE when using a different sign convention on the log_vars? The sign convention on Kendall log_vars (losses.py line 385: `self.log_vars = nn.Parameter(torch.zeros(num_tasks))`) is critical: is the actual precision computed as `exp(log_var)` or as `exp(-log_var)`?

---

### Q12: HeadPoseFiLM Receiving Signal From Converged Pose

**Context:** The `HeadPoseFiLM` module (~400K parameters: gamma_net and beta_net MLPs that modulate C5 features based on head pose) conditions the activity head's input features on head pose (model.py line ~2034: `c5_mod = self.headpose_film(c5_mod, head_pose.detach())`). The `head_pose.detach()` call stops gradient from flowing through the pose head (documented at agent4-model.md line 192: "Per paper spec"). The pose loss has been at 0.03--0.15 since epoch 2 (status file loss tables), and pose angular MAE is ~11 degrees -- within the paper target range of 8--13 degrees.

**The question:** Since `head_pose.detach()` prevents activity gradient from going through the pose head, the FiLM module's gamma/beta networks are trained ONLY by the activity gradient flowing backwards through `c5_mod -> HeadPoseFiLM` (agent6-gradient-flow.md lines 133--140). The FiLM modules receive gradient from activity's need to disentangle pose-conditioned features. But if the pose head has already converged to stable low-error predictions, the pose input to FiLM is near-constant (small frame-to-frame pose variation). This means the FiLM conditioning is computing gamma/beta from an essentially fixed input. After epoch 15, could both the pose head and HeadPoseFiLM be frozen (requires_grad=False) without reducing activity accuracy, thereby saving ~0.5M parameters of gradient compute per step?

---

## Section 4: Multi-Task Balancing (Q13--Q16)

---

### Q13: Kendall Weight Convergence Speed

**Context:** Kendall log-variances evolve slowly over ~5,700 steps (status file section 5): `lv_det` from 0.004 to 0.075, `lv_act` from 0 to 0.114, `lv_psr` from -0.001 to -0.014, and `lv_pose` permanently at -1.000. The Kendall loss formula (losses.py `MultiTaskLoss`, line 385) is: `L_total = sum_i (exp(-log_var_i) * task_loss_i + log_var_i)`. The gradient with respect to `log_var_i` is `dL/dl_i = -0.5 * exp(-l_i) * task_loss_i + 1` (there is a factor of 0.5 in the standard formulation from the original Kendall et al. paper). At equilibrium: `exp(-l_i) * task_loss_i = 2`, so precision `exp(-l_i) = 2 / task_loss_i`.

**The question:** At lv_det=0.075 and det loss ~1.0, the equilibrium condition would require `exp(-0.075) * 1.0 = 2` (false -- LHS is 0.928). For equilibrium, lv_det would need to satisfy `exp(-lv_det) * 1.0 = 2` giving `lv_det = -ln(2) = -0.693`. But lv_det is at 0.075 and RISING (moving away from -0.693). Is the Kendall optimizer's learning rate for log_vars too low (shared with model parameters at 5e-4) to reach equilibrium within 100 epochs? At the current rate (0.004 to 0.075 over 5,700 steps = ~1.25e-5 per step), reaching lv_det = 0.5 (precision = 0.606) would take ~34,000 more steps (~6 more epochs). Should the log_vars have a separate, higher learning rate (e.g., lr=0.01 instead of the shared 5e-4) to reach equilibrium faster, or is the slow evolution actually correct behavior (the task losses themselves are non-stationary as the model learns)?

---

### Q14: Combined Metric Weight Sensitivity

**Context:** The combined metric for gate evaluation is computed at evaluate.py (agent15-validation-metrics.md line 206) as: `combined = 0.30 * det_mAP50_pc + 0.35 * act_macro_f1 + 0.15 * (1/(1 + pose_fwd_MAE_deg)) + 0.20 * psr_f1_at_t`. At epoch 2: det_mAP50_pc=0.133, act_macro_f1=0.006, pose_fwd_MAE=11.32 deg, psr_f1_at_t=0.0 (from all-ones predictions producing no transitions). The combined score is: 0.30*0.133 + 0.35*0.006 + 0.15*(1/12.32) + 0.20*0.0 = 0.0399 + 0.0021 + 0.0122 + 0.0 = 0.0542. Note the discrepancy with metrics.py which uses different weighting (uniform 0.25 weights with `max(0, 1-MAE/10)` normalization) -- flagged as issue #1 in agent15-validation-metrics.md line 223.

**The question:** The combined metric gives 65% weight to det+act and 35% to pose+PSR. But PSR is structurally zero (all-ones predictions give psr_f1_at_t=0.0), and PSR only trains on 25% of batches. Pose contributes only raw `1/(1+11.32)=0.081` weighted to 0.012 (7% of the combined score). So 65% of the measured combined score comes from det+act alone. Should the gate criteria be reweighted to either (a) increase pose weight to 0.25 (from 0.15) because pose is the only genuinely converged head, or (b) reduce PSR weight to 0.05 until detach_psr_fpn is flipped at RF6+? Alternatively, should the gate use separate per-head thresholds (det >= 0.15, act >= 0.05, pose MAE < 15 deg, PSR binary acc > 0.30) rather than a single combined score? The current approach masks which head is responsible for gate failures.

---

### Q15: PSR Gradient Amplification Cascade

**Context:** PSR gradient is amplified through a cascade before Kendall weighting: `PSR_WEIGHT=10.0` (config.py `PSR_WEIGHT`, agent10-psr.md line 25) multiplies the raw PSR loss. Then `PSR_SEQ_LOSS_SCALE=1.5` (line 26) further multiplies it on seq batches. The effective PSR loss before Kendall is `PSR_raw * 10.0 * 1.5 = PSR_raw * 15.0`. After Kendall, precision `exp(-lv_psr)` at lv_psr=-0.014 is exp(0.014)=1.014 (near-identity). So PSR contributes approximately `loss * 15.0 * 1.014 = loss * 15.2` to the total loss on seq batches. On non-seq batches, PSR loss is structurally zero (transition objective disabled: `USE_PSR_TRANSITION=False`), so PSR contributes exactly 0. This creates a 15:1 swing between seq and non-seq batch PSR contributions.

**The question:** The 15x amplification creates a massive gradient magnitude swing between seq batches (PSR contributes full 15x) and non-seq batches (PSR contributes zero). This violates the Kendall assumption of stationary task losses. Is this amplification harming PSR training by: (a) creating gradient explosion on seq batches that gets clipped by `clip_grad_norm_` at 1.0 (train.py line 1622), followed by (b) 3 batches of zero PSR gradient where other heads "fill in" the feature space vacated by the clipping? Would it be more effective to (a) set `PSR_WEIGHT=1.0` (remove amplification) and let Kendall naturally balance it, or (b) keep amplification but set `PSR_LOSS_CAP` lower (e.g., 5.0 instead of 20.0) to prevent any single batch from dominating the total loss?

---

### Q16: Optimal seq_every for PSR Transformer Convergence

**Context:** The PSR causal transformer (3 layers, 4 heads, d_model=256, d_ff=1024, per agent10-psr.md line 14) processes sequences of length `PSR_SEQUENCE_LENGTH=2` (line 21: "Causal context window = 2 frames"). With `PSR_SEQ_EVERY_N_BATCHES=4` (F7 fix), PSR gets 1/4 the training updates of other heads each epoch. The status file section 6 confirms: the model uses 6,580 batches per epoch, giving ~1,645 PSR seq batches per epoch vs ~4,935 non-seq batches. Each seq batch processes PSR independently per-frame (not as a temporal sequence when `USE_PSR_TRANSITION=False`). The transformer sees exactly 2 consecutive frames per PSR sequence step, with a causal mask forcing attention only to current and one previous frame.

**The question:** With `PSR_SEQUENCE_LENGTH=2`, the transformer sees only current + 1 previous frame. For assembly phase transitions that span 10--30 frames (a worker picking up a part and placing it), this is insufficient temporal context (agent10-psr.md line 154: "A causal transformer with T=2 sees only the current frame and the immediately preceding frame. For assembly transitions spanning 5+ frames, this is insufficient temporal context."). If we increase `PSR_SEQUENCE_LENGTH` to 8--16, VRAM scales linearly (the transformer processes [B, T, 256] tensors, so T=16 uses ~8x the memory of T=2). Given current VRAM reserved is ~8.5 GB out of 16 GB (status file GPU state), there is ~7.5 GB headroom. Is it feasible to increase PSR_SEQUENCE_LENGTH to 8 or 16 without OOM on the 5060 Ti? And if so, should `PSR_SEQ_EVERY_N_BATCHES` be reduced back to 2 (since each seq batch now has 8--16x more PSR signal per batch)?

---

## Section 5: Infrastructure and Reproducibility (Q17--Q20)

---

### Q17: CUDA 13.0 + Driver 595.71.05 + Blackwell Known Incompatibility

**Context:** CUDA 13.0 is a very early major release (first CUDA 13.x). Driver 595.71.05 is at the leading edge. The RTX 5060 Ti (Blackwell) may require a minimum driver version substantially higher than consumer Ada cards. The `cudaErrorLaunchTimeout` (error code 702 on CUDA 12.x, potentially renumbered on 13.0) indicates the GPU's hardware watchdog timer (default ~2 seconds on consumer Linux cards) was not reset within the timeout window. On the v7 run (rf4_fable6_010909, PID 1300773), the training survived 8 hours 44 minutes -- the longest ever -- before timing out at epoch 5 batch ~102.

**The question:** Is there a documented incompatibility between CUDA 13.0 + driver 595.xx and Blackwell compute that causes kernel launch synchronization failures? The specific symptom (crash at epoch 5 batch ~100, never at startup) suggests a cumulative effect: memory fragmentation increasing launch latency, or GPU temperature throttle lengthening kernels as the thermal solution saturates (95% GPU utilization sustained for 3--4 hours per epoch). Would `nvidia-persistenced` (which prevents GPU driver state re-initialization between processes) help? Is there a Linux equivalent to Windows TDR registry key `TdrDelay` that can be set via `nvidia-smi -a` or a sysfs parameter to extend the watchdog from 2 seconds to 5--10 seconds?

---

### Q18: Working Around the Watchdog With the RTX 3060

**Context:** The RTX 3060 (12 GB, Ampere, driver 550 series) has never exhibited `cudaErrorLaunchTimeout` in any run. It currently drives the display (Xorg/GDM) and is invisible to CUDA compute. The RTX 5060 Ti is a secondary compute-only GPU. The timeout is per-GPU -- it only resets the affected GPU, not all GPUs. If gdm3 is stopped, the 3060 becomes available for CUDA compute. Current VRAM usage is ~8.5 GB reserved on the 5060 Ti (status file section 6).

**The question:** If we switch to the 3060 (Ampere), would the timeout disappear permanently? The 3060 has 12 GB vs 5060 Ti's 16 GB -- with ~8.5 GB currently reserved on the 5060 Ti, the 3060 at batch=4 would have ~3.5 GB headroom. Is this sufficient, or would we OOM on the 3060? The 3060 has slower memory bandwidth (360 GB/s vs ~448 GB/s on Blackwell), so step time would increase from ~1.7--2.0 seconds to perhaps ~2.2--2.6 seconds per iteration. But with no watchdogs, this could still complete epoch 100 faster than a crashing 5060 Ti. Is the Blackwell timeout likely a hardware bug that can only be worked around (not fixed) by driver updates, or is it specific to this early CUDA 13.0 release?

---

### Q19: RF5--RF10 Presets on the 5060 Ti

**Context:** The RF5--RF10 presets (defined in config.py `apply_preset()`, lines 1188--1281) use `BATCH_SIZE=6, GRAD_ACCUM_STEPS=4` (effective batch 24) -- developed for the 5060 Ti before the timeout issue was discovered. The current RF4 run uses `BATCH_SIZE=4, GRAD_ACCUM_STEPS=4` (effective batch 16) as a stability mitigation. The status file section 7 gives a "60% probability that training stabilizes through epoch 100" with current mitigations (TF32 off, V8 API disabled, batch=4). The difference in effective batch size (16 vs 24) changes batch normalization statistics and gradient noise scale.

**The question:** If the timeout does not recur, should RF5--RF10 presets be permanently set to batch=4 (from batch=6) as a conservative measure? With `ONE_CYCLE_PEAK_FACTOR=0.75` (F4 fix) and `pct_start=0.1`, the LR at peak (epoch 12) will be `BASE_LR * 0.75 = 3.75e-4`. The linear scaling rule (Goyal et al., 2017) suggests LR should scale with `sqrt(batch)`: for batch=16 vs batch=24, optimal LR would be `3.75e-4 * sqrt(16/24) = 3.06e-4`. Does this warrant a config change to `BASE_LR=4e-4` for batch=4 presets to compensate, or is the 20% difference small enough to ignore in practice?

---

### Q20: Ablation Suite Shortcut

**Context:** Fix F16 added 4 ablation presets: `det-only`, `act-only`, `psr-only`, `pose-only` (config.py, documented in status file section 1.2) plus `run_ablation_suite.sh` to orchestrate them. Running each to convergence (100 epochs, batch=4, ~3--4 hours per epoch, full dataset of ~25K frames with ~6,580 batches per epoch) would take ~300--400 hours per ablation run = ~1,200--1,600 hours for all 4 sequential runs (~50--67 days). The status file gives a 60% probability of the FULL model reaching epoch 100, meaning ablation runs may face similar or worse stability.

**The question:** Could a shorter schedule (10 epochs, 25% data subset) produce reliable head rankings? At epoch 10 (warmup nearly complete, LR ~3.75e-4 approaching peak), single-task detection should reach mAP50_pc ~0.10--0.15, single-task pose MAE ~12--15 degrees, and single-task activity should show some pred_distinct > 5. If the relative ranking of head contributions at epoch 10 (det > pose > act > psr, from the full model's epoch-2 trend) matches the final full-model ranking at epoch 100, then the short schedule is sufficient for ablation purposes. Is there theoretical justification that head rankings by validation metric stabilize by epoch 10 (before peak LR), or could a head that is silent at epoch 10 become dominant at epoch 30 (e.g., activity, which only starts learning after the ramp completes at epoch 3 and may take 20+ epochs to separate features)?

---

## Appendix: Quick-Reference Numbers

| Quantity | Value | Source |
|----------|-------|--------|
| Epoch 2 det_mAP50_pc | 0.133 | Status section 4.1 |
| Epoch 2 act_macro_f1 | 0.006 | Status section 4.1 |
| Epoch 2 pose fwd MAE | 11.32 deg | Status section 4.1 |
| Epoch 2 psr_binary_acc | 0.291 | Status section 4.1 |
| Epoch 2 combined | 0.183 | Status section 4.1 |
| Epoch 2 pred_distinct | 5/69 classes | Status section 4.1 |
| Epoch 2 entropy | 1.270 nats | Status section 4.1 |
| Activity loss epoch 2 | 0.9--1.76 | Loss table epoch 2 |
| Activity loss epoch 4 | 1.91--5.04 | Loss table epoch 4 |
| lv_det progression | 0.004 to 0.075 (5,700 steps) | Status section 5 |
| lv_act progression | 0 to 0.114 (5,700 steps) | Status section 5 |
| lv_pose | -1.000 (permanent) | Status section 5 |
| lv_psr | -0.001 to -0.014 (5,700 steps) | Status section 5 |
| GPU VRAM allocated | 1.30 GB | Status section 6 |
| GPU VRAM reserved | ~8.5 GB | Status section 6 |
| Step time | 1.7--2.0 s/it | Status section 6 |
| Epoch time | ~3--4 hours | Status section 6 |
| seq_every | 4 (F7 fix) | Status section 1.2 |
| FOCAL_ALPHA | 0.50 (F8 fix) | Status section 1.2 |
| ACT_RAMP_EPOCHS | 3 (F9 fix) | Status section 1.2 |
| ACTIVITY_HEAD_GRAD_CLIP | 5.0 (F10 fix) | Status section 1.2 |
| PSR_WEIGHT | 10.0 | agent10-psr.md line 25 |
| PSR_SEQ_LOSS_SCALE | 1.5 | agent10-psr.md line 26 |
| PSR_LOSS_CAP | 20.0 | agent10-psr.md line 27 |
| PSR_SEQUENCE_LENGTH | 2 | agent10-psr.md line 21 |
| Combined metric weights | 0.30 / 0.35 / 0.15 / 0.20 | agent15-validation-metrics.md line 206 |
| HeadPoseFiLM params | ~400K | Estimated from FiLM architecture |
| Total fixes applied | 24 (16 F-series + 8 misc) | Status section 1.2 |
| Model architecture | ConvNeXt-Base + MViTv2 + 5 heads | model.py |
| GPU count | 2 (5060 Ti for compute, 3060 for display) | Status section 6 |
