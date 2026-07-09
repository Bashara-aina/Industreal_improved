# 178 — 50 Deep Questions for the MTL Path Decision

**Purpose:** Guide Opus consultation to determine whether MTL is helping or hurting, and which path to take.
**Format:** Each question includes (a) question, (b) why it matters for the path decision, (c) evidence we have, (d) evidence missing, (e) what the answer changes.
**Provenance:** MTL optimization deep-dive based on 6 epochs of `mtl_mvit_run3` data.

---

## Section 1: The Kendall Paradox (K-1 to K-10)

### K-1: Is the Kendall weight spiral a fundamental limitation or a tuning problem?

(a) log_var_act has risen from -0.5→3.20 in 6 epochs, heading toward the +4 cap. At the cap, activity backbone gradient weight = exp(-4) = 0.018 — essentially zero. Is this behavior inevitable when per-task losses differ by 40×?

(b) If this is fundamental, Kendall is fundamentally incompatible with SEVERE loss-scale mismatches and we should replace it. If it's a tuning problem (better init, different LR), we can fix it.

(c) Evidence: log_var trajectory across epochs, loss trajectory showing activity at 12.31 vs det at 0.31. The 40× gap is consistent — activity CE naturally produces larger loss values than detection CIoU+DFL+focal.

(d) Missing: A controlled experiment with `log_var_init = [0, 0, 0, 0]` instead of -0.5. A run with higher learning rate for log_vars. A run with separate LR for log_var_act.

(e) ➤ If fundamental → Path C (fixed weights). If tuning → Path A (caps). This is THE critical question.

---

### K-2: What is the correct log_var initialization for scaled losses?

(a) Kendall (2018) initializes log_vars to 0 (σ=1), giving equal initial weights. We initialize to -0.5 (σ=0.61). Does this choice matter when activity loss starts at 4.74 vs detection at 3.85?

(b) If log_var initialization biases the trajectory, we may be able to choose inits that prevent the spiral.

(c) Evidence: Batch 0 losses — det=3.85, act=4.74, psr=1.62, pose=1.60. With log_var=-0.5: det_weight=1.65, act_weight=1.65, psr_weight=1.65, pose_weight=1.65. Weighted: det=6.35, act=7.82, psr=2.67, pose=2.64.

(d) Missing: The log_var update dynamics — how does each log_var respond to its weighted loss contribution? Does log_var_act grow because its weighted contribution is largest (7.82), or because its unweighted loss is largest (4.74)?

(e) ➤ If init matters → try log_var_init=[0.5, -1.0, 0, -0.5] to compensate for expected loss scales. 10 min experiment.

---

### K-3: Does the log_var clamping [-4, 4] create a degenerate equilibrium?

(a) Four of the six possible extreme states are: log_var=+4 (weight=0.018, task excluded), log_var=-4 (weight=54.6, task dominates). If activity hits +4, it's excluded from backbone learning. Is there a Nash equilibrium where 1-2 tasks are always excluded?

(b) The [-4, 4] range creates a 3000× dynamic range in per-task weights. This may be too wide — allowing tasks to be effectively eliminated.

(c) Evidence: log_var_act approaching +4. log_var_pose at -0.49 (stable). log_var_det at -0.41 (stable). log_var_psr slowly rising 0.65→0.94.

(d) Missing: An experiment with narrower range [-2, 2] (55× dynamic range) or [-1, 1] (7× dynamic range). Does narrower range prevent exclusion?

(e) ➤ If clamping range causes exclusion → cap the range AND add per-task caps. This is Path A with narrower global range.

---

### K-4: Is log_var_act spiraling because activity CE loss is structurally large, or because the activity head is genuinely uncertain?

(a) Kendall's theory: log_var represents aleatoric uncertainty. High uncertainty → high log_var → low weight. But activity CE loss is ~40× detection loss because CE has no upper bound (loss can be arbitrarily large for wrong predictions), while CIoU is bounded [0, 2]. Is log_var_act measuring loss-scale mismatch rather than uncertainty?

(b) This is a critical distinction. If log_var measures loss-scale, Kendall is doing loss balancing — not uncertainty weighting. If so, we should balance losses directly (scale each loss to [0, 1] range) instead of using Kendall.

(c) Evidence: Activity loss = 12.31 → sigmoid(loss) ≈ 1.0. Detection loss = 0.31. The 40× gap is structural, not uncertainty-based.

(d) Missing: A run where each loss is normalized to [0, 1] before Kendall weighting. A plot of log_var vs loss magnitude across training.

(e) ➤ If Kendall is doing loss scaling → replace with explicit loss balancing. Path C becomes the right answer.

---

### K-5: What is the PCGrad cosine conflict between activity and detection at the backbone?

(a) PCGrad projects conflicting gradients. If activity and detection gradients have cosine similarity > 0.3 (aligned), PCGrad keeps both. If < -0.3 (conflicting), it projects. What is the actual conflict angle?

(b) If activity and detection are naturally aligned (both benefit from spatial features), PCGrad may be unnecessary or harmful. If they conflict, PCGrad is essential.

(c) Evidence: We have PCGrad running but not logging per-task cosine conflicts. Zero debugging information on gradient relationships.

(d) Missing: Add `torch.cosine_similarity(grad_act, grad_det).mean()` logging to PCGrad step. Run for 100 batches and collect distribution of conflict angles.

(e) ➤ If activity-detection conflict is low (< 0.1 mean cosine) → PCGrad overhead (4× backward passes) is wasted. If conflict is high (> 0.3) → PCGrad is essential and the 0.04 weight is the only protection.

---

### K-6: Does hp_prec_cap (pose precision capped by detection precision) actually prevent pose dominance?

(a) The hp_prec_cap mechanism: `pose_precision = min(pose_log_var, det_log_var)`. This caps pose weight at detection weight. If detection log_var = -0.41 (weight=1.51), pose weight is also capped at 1.51.

(b) Evidence: pose log_var = -0.49 (weight=1.63). Since det log_var = -0.41 (weight=1.51), the cap constrains pose weight from 1.63→1.51. A small reduction but meaningful.

(c) Missing: What happens if we remove hp_prec_cap? Does pose dominate (pushing pose weight to exp(-(-4)) = 54.6)?

(d) ➤ If hp_prec_cap is critical → keep it. If pose doesn't dominate even without it → simplify and remove.

---

### K-7: Does the order of log_var updates matter for stability?

(a) log_vars are updated by SGD simultaneously with model weights. The gradient for log_var_i is: `0.5 * exp(-log_var_i) * loss_i + 1` (from the Kendall loss formula, ignoring the 0.5 factor on the loss). When loss_i is large, the gradient pushes log_var_i up strongly.

(b) Evidence: Batch 0 gradient for log_var_act ≈ 0.5 * exp(0.5) * 4.74 + 1 = 0.5 * 1.65 * 4.74 + 1 = 4.91. This pushes log_var_act UP immediately, starting the spiral from step 0.

(c) Missing: Would a lower LR for log_vars (0.001 vs 0.01) prevent the spiral? Would updating log_vars less frequently (every 10 steps) help?

(d) ➤ If log_var update frequency/LR matters → decouple log_var optimizer from model optimizer. Easy fix: different LR, different weight decay, different schedule.

---

### K-8: Is there a theoretical guarantee that Kendall finds a stable equilibrium with 4 tasks?

(a) Kendall (2018) shows results on 2-task problems (segmentation + depth). Multi-task learning theory suggests that there is no unique equilibrium for >2 tasks with conflicting gradients. The system may cycle indefinitely.

(b) We're observing log_var_act diverging, log_var_psr slowly rising, log_var_det/pose stable. This suggests a non-equilibrium state that may never converge.

(c) Evidence: 6 epochs of monotonic log_var_act increase (0.73→3.20, every epoch).

(d) Missing: Literature review of Kendall on 3+ task problems. Simulation of Kendall dynamics with synthetic loss values.

(e) ➤ If no equilibrium exists → Kendall is the wrong tool for 4 tasks. Switch to fixed weights or GradNorm.

---

### K-9: What is the true loss-scale ratio that causes log_var divergence?

(a) We need to find the threshold ratio where Kendall starts to spiral. Activity:detection = 40:1. Activity:PSR = 9.5:1. Activity:pose = 64:1.

(b) Hypothesis: Any task whose loss is >5× the mean of other tasks will spiral. If true, activity will always spiral.

(c) Evidence: Only activity is spiraling. PSR (1.30 vs det 0.31 = 4.2×) is rising slowly but not spiraling.

(d) Missing: A synthetic experiment with controlled loss ratios to find the critical ratio.

(e) ➤ If critical ratio is ~5× → activity needs explicit protection. Path A's cap at log_var=1.0 gives weight=0.37, which may still be too low if activity needs >0.5 to learn.

---

### K-10: Could gradient accumulation interact with Kendall to amplify the spiral?

(a) We use grad_accum=2 (effective batch 4). Gradients are accumulated over 2 micro-batches, then Kendall weighting is applied on the accumulated gradient. PCGrad then projects.

(b) If micro-batch 1 produces a large activity gradient and micro-batch 2 produces a small detection gradient, the accumulated gradient may have unusual directional properties that confuse Kendall.

(c) Missing: An ablation with grad_accum=1 (effective batch 2) to see if the spiral changes. A per-micro-batch log of gradient magnitudes.

(d) ➤ If grad_accum changes spiral dynamics → test grad_accum=1 or increase to grad_accum=4 (to smooth gradient directions).

---

## Section 2: PCGrad Behavior (P-1 to P-7)

### P-1: Is PCGrad's random 4-permutation projection creating a degenerate gradient?

(a) PCGrad randomly permutes the 4 tasks, then for each task projects its gradient against all previous tasks in the permutation. With 4 tasks, the last task in the permutation is projected 3 times. Does the permutation order bias which tasks dominate?

(b) If activity is always last in the permutation, its gradient is projected 3 times — potentially removing most of its signal.

(c) Evidence: The PCGrad code generates a random permutation per step. Over 24K steps, each task should be last ~25% of the time. But we don't log the permutation.

(d) Missing: Log the PCGrad permutation for 1000 steps. Check if activity is disproportionately projected.

(e) ➤ If permutation order matters → try all 24 permutations per step (expensive) or try fixed order (det→act→psr→pose).

---

### P-2: Is PCGrad on shared backbone parameters only sufficient for preventing negative transfer at the head level?

(a) PCGrad only operates on shared backbone parameters (conv_proj, blocks). Head-specific parameters are not projected. But negative transfer at the head level (head A's loss affecting head B's head-specific params) is not addressed.

(b) If heads share features through the backbone but have conflicting needs in their head-specific layers, PCGrad doesn't help there.

(c) Evidence: The code at train_step() applies PCGrad only to `shared_backbone_params`.

(d) Missing: An analysis of head-specific parameter gradients. Do they conflict?

(e) ➤ If head-level conflict is significant → extend PCGrad to all shared parameters including FPN.

---

### P-3: What is the computational cost of PCGrad (4 extra backward passes) vs the benefit?

(a) PCGrad requires: forward pass → backward_task_1 → backward_task_2 → ... → backward_task_4 → sum projected grads → optimizer step. This is 5 backward passes instead of 1.

(b) Evidence: Our FPS is 10.97. Without PCGrad, we'd expect ~15-18 FPS (3-4 backward passes saved, each ~20% of forward time).

(c) Missing: A direct ablation — 100 batches with PCGrad vs 100 batches without. Measure FPS and loss trajectory.

(d) ➤ If PCGrad costs >30% throughput with minimal benefit → disable PCGrad and rely on Kendall alone. This is Path C-light.

---

### P-4: Does PCGrad actually change the gradient direction for any task?

(a) PCGrad only projects if gradients conflict (cosine similarity < 0). If all tasks have naturally aligned gradients (all pointing toward "extract better spatial features"), PCGrad does nothing.

(b) Evidence: We can test this by running 100 batches with PCGrad enabled but logging the cosine similarities. If mean alignment > 0.5, PCGrad is a no-op.

(c) Missing: The `torch.cosine_similarity` measurement that would answer this directly.

(d) ➤ If PCGrad is a no-op → remove it. Save 4 backward passes per step. Throughput increases ~40%.

---

### P-5: Is PCGrad's detach-from-pool mechanism correctly preventing FPN features from receiving conflicting gradients?

(a) The detection head uses FPN features. Other heads may not. If PCGrad projects activity gradient against detection gradient on shared backbone params, but activity doesn't use FPN, the projection may be based on incorrect assumptions about gradient sources.

(b) The gradient from activity head only flows through backbone → class token. The gradient from detection head flows through backbone → FPN → detection head. The shared portion is the backbone. PCGrad correctly operates there.

(c) Evidence: Implementation analysis confirms PCGrad operates on `shared_backbone_params` which includes conv_proj and blocks — the only truly shared parameters.

(d) Missing: Verification that FPN params are excluded from PCGrad.

(e) ➤ If FPN params have conflicting gradients with backbone → they should be included in PCGrad. If they're already excluded, this is correct.

---

### P-6: Does the PCGrad projection strength (magnitude after projection) vary across tasks?

(a) PCGrad projects conflicting components out, then sums the projected gradients. The resulting gradient magnitude for each task depends on how much was projected away. A task that conflicts heavily with others may have near-zero gradient after projection.

(b) This is a second mechanism (beyond Kendall weighting) that can starve a task.

(c) Missing: Per-task gradient L2 norm before and after PCGrad projection.

(d) ➤ If activity gradient magnitude drops significantly after PCGrad (beyond the Kendall weight reduction) → PCGrad is compounding the starvation. This strengthens the case for Path C (remove both Kendall and PCGrad).

---

### P-7: Could we replace PCGrad with uncertainty-weighted gradient scaling (no projection)?

(a) An alternative: scale each task's gradient by its Kendall weight, sum them without projection, and take the optimizer step. This is simpler, cheaper (1 backward pass), and doesn't project away signal.

(b) The risk: no protection against conflicting gradient directions.

(c) Missing: A 100-batch comparison of PCGrad vs weighted-sum without projection. Measure backbone parameter angle difference.

(d) ➤ If weighted-sum without projection produces similar gradient directions → remove PCGrad. Path C becomes: remove PCGrad, keep Kendall (with caps).

---

## Section 3: Activity Head Starvation (A-1 to A-7)

### A-1: Is a LayerNorm→Linear head sufficient for activity recognition with MViTv2-S features?

(a) The activity head is embarrassingly simple: 768→75 linear layer with LayerNorm input. No temporal modeling. Compare to WACV SOTA MViTv2-S which uses the full video-level classification head (768→75 with temporal pooling).

(b) Even with full backbone gradient, can this head achieve >50% top-1?

(c) Evidence: Single-task MViTv2 on Kinetics-400 uses a learned class token + FC head. Our head is even simpler.

(d) Missing: A single-task activity MViTv2-S run to establish the ceiling for our head architecture.

(e) ➤ If head architecture limits top-1 to <30% even at single-task → we need a better head regardless of MTL fix. Activity head redesign should happen in parallel with Path A.

---

### A-2: Does the activity head's 0.04 Kendall weight explain all of its poor performance?

(a) If activity had weight=1.0 (same as detection), how much would its backbone gradient increase? Currently: grad_act = 0.04 × ∇θ_L_act. With weight=1.0: grad_act = 1.0 × ∇θ_L_act. **25× more gradient.**

(b) Evidence: Activity top-1 = 0.008 (epoch 5 eval). Random on 75 classes = 0.0133. Activity is below random. This is consistent with near-zero backbone gradient.

(c) Missing: The counterfactual — what is activity performance after 6 epochs with weight=1.0?

(d) ➤ If activity at weight=0.04 has top-1=0.008, and random=0.0133, then even restoring weight might only get to ~0.10-0.20 (still far from SOTA 0.65). The head architecture may still be the bottleneck.

---

### A-3: Does the class weight imbalance (72 nonzero of 75 classes, weights range 0-137) interact with Kendall to suppress low-frequency classes?

(a) Activity uses inverse-frequency class weights. Some classes have weight > 100× others. The weighted CE loss can be dominated by rare-class mistakes.

(b) If rare classes dominate the loss, Kendall sees a high loss and increases log_var_act — even though the head may be performing well on common classes.

(c) Evidence: class weight min=0.0 max=137.2 mean=9.98. 3 classes with weight=0.

(d) Missing: Decompose activity loss by class frequency bin. Are rare classes driving the 12.31 loss?

(e) ➤ If rare classes inflate activity loss → use class-weight normalization (divide weighted loss by sum of weights) or use focal loss for activity. This would reduce the activity loss magnitude and potentially stop the Kendall spiral.

---

### A-4: Could label smoothing (0.1) be causing the activity head to never produce confident predictions?

(a) Label smoothing with ε=0.1 replaces hard targets with [0.9 for correct, 0.1/74 for incorrect]. This prevents the head from ever producing 100% confident predictions, which increases the minimum achievable loss.

(b) With 75 classes and ε=0.1: minimum loss = -0.9*log(0.9) - 0.1*log(0.1/74) ≈ 0.095 + 0.1*log(740) ≈ 0.095 + 0.66 ≈ 0.755 per sample. Times 75 classes weighted by frequency.

(c) Evidence: We use label_smoothing=0.1 in activity_loss().

(d) Missing: An ablation with label_smoothing=0.0. Would activity loss drop from 12.31 to ~5-8?

(e) ➤ If label smoothing inflates loss → reducing smoothing (or removing it) lowers activity loss, which reduces the Kendall spiral pressure. Quick test: change to 0.05 or 0.0.

---

### A-5: Is the activity head learning anything at all, or is it stuck at random?

(a) Activity performs slightly below random (0.008 vs 0.0133). But it consistently predicts class 11 — the most common class. This is better than random guessing (which would spread predictions across 75 classes). The head has learned the data distribution but not class discrimination.

(b) Evidence: All 4 diagnostic samples predict class 11 at ~7% confidence. Class 11 is likely the most frequent class in the training set.

(c) Missing: Confirmation of class 11 frequency in training data. Per-class precision/recall from epoch 5 eval.

(d) ➤ If the head learned the prior distribution but not discrimination → it is technically learning, just very slowly. With 25× more gradient (weight=1.0), it would likely start discriminating. This supports Path A.

---

### A-6: Would removing the activity head's weight from the Kendall loss (treating it as a fixed-weight auxiliary task) improve overall MTL performance?

(a) Alternative: Keep activity in the model (forward pass, predictions) but remove it from Kendall weighting. Use a fixed weight for activity's contribution to backbone gradient: `backbone_loss = det_loss + psr_loss + pose_loss + 0.5 * act_loss` (fixed weight, no learned log_var).

(b) This gives activity a steady gradient signal while preventing its large loss from corrupting Kendall for other tasks.

(c) Evidence: Activity dominates Kendall's total loss but gets minimal weight. Removing it from Kendall would give other tasks more balanced weights.

(d) Missing: The effect on other tasks' log_vars when activity is excluded from Kendall.

(e) ➤ If activity's Kendall exclusion stabilizes other tasks → we should detach activity from Kendall. This is a hybrid approach: Kendall for det/psr/pose, fixed weight for activity.

---

### A-7: What is the optimal activity weight given the loss scale mismatch?

(a) If we fix activity weight to a constant (no Kendall), what weight maximizes overall MTL performance? Too high → activity dominates backbone, detection/PSR degrade. Too low → activity doesn't learn.

(b) Evidence: activity loss = 12.31, detection loss = 0.31. A weight ratio of 0.31/12.31 ≈ 0.025 would equalize their contributions. Kendall is at 0.04 — close to this ratio.

(c) Missing: A sweep of fixed activity weights: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]. Run each for 1 epoch and measure activity top-1.

(d) ➤ The "optimal" weight for loss equalization (~0.025) may be too low for learning to happen. Activity may need 0.25-0.5 to learn, even though this would dominate the total gradient signal. This is the core tension: what's "fair" vs what's "enough."

---

## Section 4: PSR Temporal Collapse (S-1 to S-7)

### S-1: Is PSR flat because of Kendall suppression, because of the detached FPN, or because of the causal Transformer architecture?

(a) Three possible causes: (1) Kendall weight=0.39 is too low, (2) DETACH_PSR_FPN prevents gradient flow to backbone, (3) the causal TransformerEncoder on top of pooled features cannot learn temporal PSR patterns.

(b) Each cause has a different fix. Cause 1 → Path A. Cause 2 → remove detach. Cause 3 → redesign PSR head.

(c) Evidence: PSR loss flat at 1.30 across all 6 epochs. log_var_psr rising 0.65→0.94. DETACH_PSR_FPN=True in config.

(d) Missing: An ablation removing DETACH_PSR_FPN (allows PSR gradient to flow to backbone). A diagnostic that checks whether PSR head parameters are changing at all.

(e) ➤ If the detach is the main cause → remove it (simple change). If the transformer architecture is the cause → need redesign. If Kendall weight is the cause → Path A helps.

---

### S-2: Is DETACH_PSR_FPN preventing PSR gradient from ever reaching the backbone?

(a) DETACH_PSR_FPN=True means the PSR head's input features are detached from the computation graph. No gradient flows from PSR loss to backbone. The backbone never receives PSR-specific feature shaping.

(b) This was added to prevent PSR from corrupting FPN features (which are shared with detection). But it also means PSR cannot influence the backbone — it can only train its head parameters.

(c) Evidence: Config has DETACH_PSR_FPN=True. PSR loss is flat, suggesting no learning.

(d) Missing: A run with DETACH_PSR_FPN=False. Expected: PSR loss decreases, but detection mAP may also decrease if gradient conflict exists.

(e) ➤ If detach is the cause → remove it and accept potential detection mAP cost. Tradeoff: PSR improves, detection may degrade slightly.

---

### S-3: Does PSR's causal TransformerEncoder actually receive useful temporal features?

(a) The PSR head uses AdaptiveAvgPool3d on conv_proj features → [B, 96, T=8]. These are early-layer features (conv_proj is the first layer). Do conv_proj features contain temporal PSR information?

(b) Conv_proj is a learned 3D convolution projecting 3-channel input to 96 channels with temporal stride 2 (T=16 → T=8). It captures low-level motion but not high-level procedure steps.

(c) Evidence: Diagnostic shows PSR per-component predictions near 0.5 for most components — barely above random.

(d) Missing: A comparison of PSR using conv_proj features vs block-14 (class token) features. The class token at block 14 may contain more temporal context.

(e) ➤ If conv_proj features are insufficient → change PSR input to later backbone features. MViTv2's later blocks have better temporal modeling.

---

### S-4: Is the 3-layer causal Transformer encoder adequate for PSR, or does it need more capacity?

(a) 3 transformer layers with nhead=4, d_model=96. Total transformer params ≈ 3 × (4 × 96² + 2 × 96²) ≈ 3 × 55K ≈ 165K. The full PSR head is larger but this is the core reasoning component.

(b) PSR requires understanding procedure step transitions — which can span 30+ frames. A 3-layer transformer with d_model=96 operating on T=8 tokens may lack capacity.

(c) Missing: An ablation with 6 layers (double capacity) or d_model=192 (double width).

(d) ➤ If transformer capacity is the bottleneck → increase capacity. PSR head parameters are only ~0.7M of 43.5M total — increasing to 1.5M is negligible.

---

### S-5: Why does every PSR component have mean prediction near 0.5-0.7 except component 0?

(a) Components 3-10 have mean sigmoid activation ~0.5-0.7 (uncertain). Component 0 has mean 0.008 (almost never predicted). This suggests the PSR head is predicting a narrow range of activation values regardless of the true label.

(b) This is consistent with the PSR head receiving backbone features that don't contain PSR-relevant information — it learns per-component biases (comp 0 → rarely positive, others → ~0.6) and doesn't react to input.

(c) Evidence: Per-component means from epoch 6 diagnostics. Comp 0 mean=0.008, comps 1-2 mean=0.45-0.47, comps 3-10 mean=0.52-0.68.

(d) Missing: Per-component prediction variance (std across time/frames). If variance is near zero, the head is pure bias.

(e) ➤ If predictions are pure bias → PSR head is not using input features at all. This supports the detach/Kendall starvation hypothesis. Path A or removing detach would restore feature utilization.

---

### S-6: Does PSR's loss trajectory match ConvNeXt's PSR collapse pattern exactly?

(a) The previous ConvNeXt-based MTL model also showed flat PSR loss. The fix then was replacing GELU with LeakyReLU in the Transformer encoder. We applied that fix, but PSR is still flat.

(b) This suggests the GELU fix was necessary but not sufficient. The remaining issue is either Kendall suppression, detach, or both.

(c) Evidence: Previous run PSR loss trajectory (flat). Current run PSR loss trajectory (flat). Same pattern.

(d) Missing: A ConvNeXt PSR loss trajectory plot for side-by-side comparison.

(e) ➤ If the same pattern repeats → the root cause is not architecture-specific but optimization-specific (Kendall + detach). Path A directly addresses this.

---

### S-7: Could PSR benefit from a classification head instead of per-frame BCE?

(a) PSR is framed as 11 binary classification problems (BCE per component). An alternative is a 11-class classification head that predicts which component is active at each frame (single-label classification).

(b) Single-label classification would reduce output dimension (11 outputs instead of 11 independent sigmoids) and may be easier to learn. It also allows CE loss which has stronger gradients than BCE for confident wrong predictions.

(c) Evidence: Multi-label BCE produces flat gradients for well-separated features. CE produces stronger gradients for wrong classifications.

(d) Missing: Prototype of a single-label PSR head. Would need to handle multi-component frames (when multiple components are active).

(e) ➤ If BCE is structurally weak for PSR → switch to CE with multi-label head (11-class softmax per frame). This is an architecture change independent of MTL optimization.

---

## Section 5: Detection Class Discrimination (D-1 to D-6)

### D-1: Is detection class-0 collapse a normal precursor to class discrimination, or a sign that class discrimination will never start?

(a) Every cell predicts class 0 at 0.9999 confidence. This is the "background" or "objectness" class. In YOLOv8 training, class discrimination typically starts at epochs 8-15, after the model learns objectness.

(b) Our epoch 5 eval shows no class discrimination. This may be normal (we're at epoch 6, class discrimination starts at 8-15), or it may be caused by Kendall/PCGrad.

(c) Evidence: YOLOv8 literature: class discrimination emerges at epochs 8-15. Our epoch 5 eval is consistent with this timeline.

(d) Missing: Epoch 10 eval data (next eval point). If class discrimination starts by epoch 10, it's normal. If not, MTL is the cause.

(e) ➤ If class discrimination starts normally at epoch 8-15 → no action needed. If it remains at epoch 10 → detection class discrimination is being suppressed by MTL, not just normal training dynamics.

---

### D-2: Does detection benefit from shared backbone features (compared to single-task)?

(a) The MTL hypothesis claims detection benefits from activity/PSR/pose features. Does it? Or would a single-task detection MViTv2-S achieve higher mAP with the same compute?

(b) Evidence: Detection loss trajectory is healthy (0.19→0.31). DFL decode works. CIoU is active. But mAP=0.0 at epoch 5 (class collapse, normal).

(c) Missing: Single-task detection MViTv2-S baseline. This is the most important ablation for the MTL hypothesis.

(d) ➤ If single-task detection mAP > MTL detection mAP at any epoch → MTL is hurting detection. If MTL detection matches or exceeds single-task → MTL is helping (or at least not hurting). This directly tests the hypothesis.

---

### D-3: Do FPN features degrade because PSR gradient (if un-detached) would compete for early-layer features?

(a) PSR uses conv_proj features (first layer). Detection uses multi-scale features from FPN (conv_proj through blocks 1, 3, 14). If PSR gradient flows to conv_proj, it modifies early features that detection also depends on.

(b) This is why DETACH_PSR_FPN=True exists — to protect detection features from PSR corruption. But this also prevents PSR from learning.

(c) Evidence: FPN design creates shared low-level features that both detection and PSR need.

(d) Missing: Gradient cosine similarity between PSR head gradient (on conv_proj) and detection head gradient (on conv_proj). If they conflict > 0.3, detach is justified.

(e) ➤ If PSR-detection gradient conflict is high → detach is necessary, and PSR needs an alternative gradient path (like an auxiliary loss on later features). If conflict is low → remove detach.

---

### D-4: Is the 24-class detection task too fine-grained for early MTL training?

(a) 24 assembly-state classes include fine-grained distinctions (e.g., "screw-in-progress" vs "screw-complete"). These require subtle visual differences that may conflict with activity classes.

(b) Detection may need more epochs (20-30) before class discrimination starts because the task is inherently harder.

(c) Evidence: Class 0 (likely "background" or "object") dominates all predictions at epoch 5.

(d) Missing: Per-class training set label distribution. What is the most common detection class? Is class 0 truly background?

(e) ➤ If detection class discrimination naturally requires 20+ epochs → the model needs longer training regardless of MTL. The epoch 10 eval is critical.

---

### D-5: Does the FPN's 256-channel bottleneck limit detection capacity?

(a) LightweightFPN projects 96→256, 192→256, 384→256, 768→256, then applies 3×3 convs. The 256-channel bottleneck may limit the FPN's ability to represent 24 detection classes.

(b) Evidence: 256 channels is standard for YOLOv8-style FPNs (YOLOv8m uses 256-channel FPN). But YOLOv8m has 25.9M params vs our FPN with fewer.

(c) Missing: FPN parameter count breakdown.

(d) ➤ If FPN capacity is the bottleneck → increase FPN channels to 384 or 512. But this also increases VRAM and GFLOPs.

---

### D-6: Is the detection NMS configuration (score_thresh, iou_thresh) appropriate for our evaluation?

(a) C.DET_EVAL_SCORE_THRESH = 0.001 (very low). C.DET_EVAL_NMS_IOU_THRESH = 0.65 (moderate). With threshold 0.001, we keep almost all boxes (4165→~400 after NMS at score>0.001).

(b) If the threshold is too low, NMS keeps noisy boxes that suppress correct boxes via IoU competition.

(c) Evidence: 4165 cells × 4 levels, ~400 after NMS at score>0.001.

(d) Missing: mAP@0.5 at different score thresholds (0.001, 0.01, 0.05, 0.1). Compare NMS output quality.

(e) ➤ If threshold sensitivity is high → tune NMS parameters. If mAP is robust to threshold changes, this is not a concern.

---

## Section 6: Architecture & MTL Design (R-1 to R-6)

### R-1: What is the optimal number of tasks for MTL with MViTv2-S?

(a) We have 4 tasks sharing one backbone. Is 4 tasks the right number? Would 3 tasks be better (dropping activity, which has the most conflicting gradient)? Would 2 tasks be optimal (detection + pose)?

(b) There's no free lunch: more tasks → more parameter sharing → less capacity per task. The MTL sweet spot depends on task relatedness.

(c) Evidence: Activity is starved (conflicting), detection is thriving, pose is stable, PSR is flat.

(d) Missing: A systematic comparison of 2-task, 3-task, and 4-task configurations.

(e) ➤ If 4 tasks fundamentally exceed the capacity of MViTv2-S's shared features → consider 3-task (detection + pose + PSR, drop activity) or 2-task (detection + pose). This is a strategic architecture decision.

---

### R-2: Should we use separate batch normalization for each task output?

(a) Currently, all heads share the same backbone features without task-specific normalization. Task-specific BatchNorm or LayerNorm on shared features could reduce negative transfer.

(b) Adding task-specific normalization at the head input would allow each head to adapt backbone features to its own distribution.

(c) Evidence: Activity head uses LayerNorm at input. Other heads don't.

(d) Missing: An ablation adding LayerNorm to detection and PSR head inputs.

(e) ➤ If task-specific normalization improves per-task performance → easy addition. Low-risk, potentially high-reward.

---

### R-3: Is MViTv2-S the right backbone for 4-task MTL?

(a) MViTv2-S is a video transformer with 34.5M pretrained params. It's designed for Kinetics-400 video classification. Is it optimal for simultaneous detection+activity+PSR+pose?

(b) Alternative backbones: Video-Swin-T (3D window attention, may be better for detection), X3D-M (lighter, faster), or even a 3D ConvNeXt variant.

(c) Evidence: MViTv2-S has hierarchical features with spatial downsampling, making it good for FPN-based detection. But its temporal pooling (T=16→T=8 at conv_proj) may discard temporal info needed for PSR.

(d) Missing: A comparison of FPN feature quality from MViTv2-S vs other backbones.

(e) ➤ If MViTv2-S is suboptimal for MTL → consider backbone change. This is a MAJOR change (weeks of work). Only justified if MViTv2-S is fundamentally incompatible with our task set.

---

### R-4: Should activity and PSR share a secondary temporal feature extractor?

(a) Both activity (75-class per-frame) and PSR (11-component per-frame) need temporal context. Currently, the activity head is spatial-only (LayerNorm→Linear on class token) and PSR has its own transformer on conv_proj features.

(b) A shared temporal feature extractor (a common transformer on backbone features) with task-specific output heads could benefit both tasks and reduce total compute.

(c) Evidence: Activity performs below random. PSR is flat. Both need temporal features they don't currently receive.

(d) Missing: A prototype architecture with shared temporal transformer for activity+PSR.

(e) ➤ If a shared temporal module improves both → major architectural win. But adds complexity. Consider for Path A post-fix.

---

### R-5: Is the class token (block 14 output) the right feature source for activity and pose?

(a) Activity and pose heads use the class token from block 14 (768-dim). The class token is a learned position used for video-level classification in Kinetics-400. It may encode video-level semantics (scenes, objects) rather than per-frame action or pose.

(b) Alternative: use spatial features from intermediate blocks (like detection uses FPN features) with global pooling per frame.

(c) Evidence: Activity at 0.008 top-1, pose at 0.19 loss (reasonable). The class token may be adequate for pose (a continuous regression problem) but insufficient for activity (a fine-grained classification problem).

(d) Missing: An ablation using pooled spatial features instead of class token for activity and pose.

(e) ➤ If class token is insufficient → change activity/pose input features. This could transform activity performance regardless of Kendall weight.

---

### R-6: Does head warmup (250 steps) actually help or hurt?

(a) Head warmup zeroes head gradients for 50 steps, then linearly ramps over 200 steps. This is designed to let the backbone stabilize before heads start influencing it. But during warmup, the backbone is training without head-specific gradients.

(b) If activity head starts with near-zero backbone gradient, and warmup further delays its learning, the log_var spiral may begin before activity ever gets a meaningful gradient.

(c) Evidence: log_var_act starts at -0.5 and begins rising immediately. By step 50 (when head gradients activate), activity has already "lost" the initialization lottery.

(d) Missing: A run with zero head warmup. Does removing warmup allow activity to get an early foothold?

(e) ➤ If warmup allows log_var_act to begin its spiral → shorten or remove warmup. Early learning signal may set log_var_act on a better trajectory.

---

## Section 7: Proving the MTL Hypothesis (H-1 to H-7)

### H-1: What is the minimum experimental design to falsify "MTL hurts"?

(a) We need to define what "MTL hurts" means empirically. Candidate definitions:
   - Definition 1: Per-task eval metric < single-task baseline for any task
   - Definition 2: Per-task eval metric < single-task baseline for ALL tasks
   - Definition 3: Sum of normalized metrics < sum of single-task normalized metrics
   - Definition 4: Training time to reach threshold metric > single-task training time

(b) The definition determines what we need to measure. Definition 1 is easiest to falsify (any degradation = MTL hurts). Definition 4 is hardest (requires threshold metrics).

(c) Evidence: We have MTL metrics at epoch 5. No single-task baselines exist.

(d) Missing: Agreement on the definition of "MTL hurts." This is a research question, not a measurement question.

(e) ➤ The definition determines the experimental design for file 179. Without a clear definition, we can't design the right experiment.

---

### H-2: For a paper claiming "MTL provides an efficient multi-task system," what is the minimum acceptable per-task performance?

(a) If the paper's claim is "one model for all tasks at reasonable accuracy," what is "reasonable"? 80% of SOTA? 60%? Equal to single-task?

(b) Without a target, we can't evaluate whether MTL "works."

(c) Evidence: Current per-task performance: activity 0.01% of SOTA, detection 0%, PSR 0%, pose ~50% of claimed SOTA.

(d) Missing: A reference point. What accuracy level makes a multi-task system practically useful for assembly monitoring?

(e) ➤ Target setting determines whether Path A (fix optimization) or a complete architecture redesign is needed.

---

### H-3: Does MTL provide better generalization than single-task models?

(a) Theoretical MTL benefit: shared features generalize better because they're shaped by multiple tasks. Empirical: MTL models often generalize worse because the feature space must serve all tasks.

(b) Evidence: Our MTL model generalizes poorly (activity below random, PSR flat). But this may be optimization failure rather than generalization failure.

(c) Missing: A comparison of MTL vs single-task generalization gap (train vs val performance).

(d) ➤ If MTL generalizes better (smaller train-val gap) → MTL benefit is real even if absolute metrics are lower. If MTL generalizes worse → MTL is fundamentally harming the model.

---

### H-4: What is the efficiency-accuracy Pareto frontier for MTL vs single-task?

(a) Each configuration (MTL, single-task per head) has an efficiency (params, FLOPs, training time) and accuracy profile. The Pareto frontier shows which configurations are not dominated.

(b) If MTL is more efficient (lower total params) AND has competitive accuracy (within 80% of single-task), it's Pareto-optimal.

(c) Evidence: MTL efficiency: 43.5M params, 129.59 GFLOPs, 73 hours for 100 epochs.

(d) Missing: Single-task efficiency: params, FLOPs, training time. Estimate: 4 × 34.5M backbone ≈ 138M params for 4 separate models. Each model needs its own training run (~18 hours each).

(e) ➤ If MTL is Pareto-optimal → the paper has a defensible efficiency contribution even with lower accuracy. This reframes from "beating SOTA" to "efficient multi-task."

---

### H-5: Can we prove MTL helps detection specifically?

(a) The strongest MTL claim would be: detection mAP is HIGHER in MTL than single-task because activity+PSR+pose provide complementary features.

(b) We need: (1) single-task detection mAP, (2) MTL detection mAP at same epoch, (3) evidence that MTL features encode cross-task information.

(c) Missing: Everything. No single-task detection baseline exists.

(d) ➤ If MTL detection mAP > single-task detection mAP → strong MTL evidence. If equal → MTL doesn't hurt detection. If lower → MTL hurts detection. This single ablation is the most important experiment in file 179.

---

### H-6: What role does the dataset size play in MTL effectiveness?

(a) MTL is most effective when per-task data is limited (tasks share data to compensate). Our dataset: 78K training windows across all tasks. Per-task data: all 78K (each window has all 4 labels).

(b) With 78K training samples per task, we have moderate data. MTL's benefit should be in feature sharing, not data augmentation.

(c) Evidence: Activity uses CE with class weights — 3 classes have zero training samples. MTL cannot compensate for missing classes.

(d) Missing: Per-class sample counts for all 4 tasks. Data efficiency analysis (how many samples needed to reach X% accuracy).

(e) ➤ If data is the bottleneck → MTL may not help as much as data augmentation or synthetic data. Consider adding data augmentation (MixUp, CutMix, frame masking).

---

### H-7: What would convince a skeptical reviewer that MTL helps?

(a) AAIML reviewers will be skeptical of MTL claims because: (1) MTL papers often show worse per-task metrics, (2) MTL optimization is notoriously hard, (3) the community has seen many "MTL helps" claims that don't replicate.

(b) A convincing claim needs: (1) single-task baselines for ALL tasks, (2) MTL model matches or exceeds single-task on at least ONE task, (3) MTL is within 80% on other tasks, (4) total MTL params < 60% of sum of single-task params, (5) total training time < 50% of sum of single-task time.

(c) Evidence: We can satisfy (4) already (43.5M vs ~100M). We need (1)-(3) for the paper. (5) may be achievable (~73 hours vs ~18×4=72 hours — roughly equal).

(d) Missing: All single-task baselines.

(e) ➤ The reviewer's bar defines our experimental requirements. File 179 should be designed to meet this bar.

---

## Summary Index — 50 Questions

| ID | Section | Question | What Answer Changes |
|----|---------|----------|---------------------|
| K-1 | Kendall | Is the spiral fundamental or tuning? | Path A vs Path C |
| K-2 | Kendall | Correct log_var init? | Change init values |
| K-3 | Kendall | Does [-4,4] clamp cause exclusion? | Narrow clamp range |
| K-4 | Kendall | Uncertainty or loss scaling? | Replace Kendall with balancing |
| K-5 | Kendall | PCGrad conflict angle? | Keep or remove PCGrad |
| K-6 | Kendall | hp_prec_cap effectiveness? | Keep or remove cap |
| K-7 | Kendall | Log_var update order/stability? | Decouple log_var optimizer |
| K-8 | Kendall | Existence of 4-task equilibrium? | Replace Kendall or not |
| K-9 | Kendall | Critical loss ratio for spiral? | Activity protection level |
| K-10 | Kendall | Grad accum × Kendall interaction? | Change grad_accum |
| P-1 | PCGrad | Permutation bias? | Fix or randomize permutation |
| P-2 | PCGrad | Head-level conflict? | Extend PCGrad scope |
| P-3 | PCGrad | Cost vs benefit? | Keep or remove PCGrad |
| P-4 | PCGrad | Does PCGrad change anything? | Remove if no-op |
| P-5 | PCGrad | Detach-from-pool correctness? | Fix if wrong |
| P-6 | PCGrad | Per-task gradient magnitude after projection? | Compound starvation |
| P-7 | PCGrad | Replace with weighted-sum? | Simpler alternative |
| A-1 | Activity | Head architecture ceiling? | Redesign if needed |
| A-2 | Activity | 0.04 weight explains all? | Fix weight |
| A-3 | Activity | Class weights inflating loss? | Normalize weights |
| A-4 | Activity | Label smoothing inflating loss? | Reduce smoothing |
| A-5 | Activity | Head learning anything? | Validate direction |
| A-6 | Activity | Remove from Kendall? | Hybrid approach |
| A-7 | Activity | Optimal fixed weight? | Set explicit weight |
| S-1 | PSR | Kendall or detach or architecture? | Root cause identification |
| S-2 | PSR | DETACH_PSR_FPN effect? | Remove detach |
| S-3 | PSR | Conv_proj features sufficient? | Change features |
| S-4 | PSR | Transformer capacity? | Increase size |
| S-5 | PSR | Per-component bias-only? | Pure bias = not using features |
| S-6 | PSR | Same as ConvNeXt collapse? | MTL-specific fix |
| S-7 | PSR | BCE vs CE head? | Architecture change |
| D-1 | Detection | Normal class collapse or MTL-stunted? | Confirm at epoch 10 |
| D-2 | Detection | Does MTL help detection? | Core hypothesis test |
| D-3 | Detection | PSR-detection feature conflict? | Justify or remove detach |
| D-4 | Detection | 24 classes too many for MTL? | Task pruning |
| D-5 | Detection | FPN 256-channel bottleneck? | Increase channels |
| D-6 | Detection | NMS config appropriate? | Tune parameters |
| R-1 | Architecture | Optimal task count? | Drop tasks |
| R-2 | Architecture | Task-specific normalization? | Add norm |
| R-3 | Architecture | MViTv2-S right backbone? | Major change |
| R-4 | Architecture | Shared temporal module? | Architecture change |
| R-5 | Architecture | Class token adequate? | Change features |
| R-6 | Architecture | Warmup helpful? | Remove warmup |
| H-1 | Hypothesis | Falsification definition? | Experimental design |
| H-2 | Hypothesis | Minimum acceptable performance? | Target setting |
| H-3 | Hypothesis | MTL generalization benefit? | Evidence direction |
| H-4 | Hypothesis | Pareto frontier? | Efficiency claim |
| H-5 | Hypothesis | MTL helps detection? | Core evidence |
| H-6 | Hypothesis | Dataset size effect? | Data augmentation |
| H-7 | Hypothesis | Reviewer threshold? | Experimental requirements |

**Total: 50 questions across 7 sections.** Use file 177 for background, file 179 for the experimental design that answers these questions, and file 180 for the Opus consultation prompt.
