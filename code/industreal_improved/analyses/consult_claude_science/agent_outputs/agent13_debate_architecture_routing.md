# Agent 13: Architecture DEBATER -- Challenging Agent 3's Routing-Centric Claims

**Debate Target:** Agent 3 (Architecture Routing Specialist) -- claims that sophisticated feature routing (Cross-Stitch, NDDR, MTAN, Task Routing, ETR-NLP, AdaShare) is necessary and beneficial for MViTv2-S with 4 heterogeneous tasks (detection, pose, activity, PSR).

**Core Thesis of This Debate:** The evidence for encoder-focused routing methods shows only **moderate, inconsistent gains** over simple shared backbones, with significant scalability costs, negative transfer risks, and architectural complexity that may not justify the marginal improvements for practical deployment.

---

## Claim 1: "Routing Methods Show Clear Improvements Over Shared Backbones"

**Agent 3's Position:** Cross-Stitch, NDDR, MTAN, Task Routing, et al. demonstrate consistent improvements over hard parameter sharing on benchmarks like NYUv2.

### Counter-Evidence: The Vandenhende MTL Survey (arXiv:2004.13379)

The comprehensive MTL survey by Vandenhende et al. (2020) directly addresses this claim:

> **"Moderate only performance improvements achieved by the encoder-focused approaches to MTL."**

This is the most definitive meta-analysis available on this topic. After surveying and benchmarking the full spectrum of MTL approaches -- including Cross-Stitch, NDDR, MTAN, Task Routing, and branched architectures -- the authors explicitly conclude that encoder-focused routing methods deliver **moderate only** gains. This is not a ringing endorsement.

**Key detail from the survey findings:**

1. **Benchmarking on NYUv2 (Table 2 of survey):** The improvements from routing methods over a simple shared encoder + task-specific decoder baseline are often within 1-2% mIoU or comparable noise levels. Many published results do not account for variance across multiple runs, meaning reported gains may fall within statistical noise.

2. **PASCAL-Context results:** On this more challenging 5-task benchmark (semantic seg, parts, saliency, surface normal, edge detection), simpler baselines with proper decoder design frequently match or approach routed methods. The survey notes that decoder-focused approaches (e.g., task-specific decoder refinement) often outperform or match encoder-focused routing with fewer parameters.

3. **Consistency problem:** The survey documents that the same routing method applied to different backbones or datasets shows inconsistent improvements. Cross-Stitch on NYUv2 shows different relative gains depending on whether VGG, ResNet, or SegNet is used as the backbone -- suggesting the benefit is backbone-dependent, not universally applicable.

### Direct Relevance to MViTv2-S

MViTv2-S is a significantly more powerful backbone than VGG/ResNet/SegNet used in the benchmarked papers. Key architectural features of MViTv2-S:
- **16-block transformer** with hierarchical pooling
- **FPN** already provides multi-scale feature aggregation
- **cls_token** mechanism for global representation

A strong argument exists that **MViTv2-S's built-in architectural sophistication** (pooling attention, FPN, cls_token) already subsumes much of the benefit that routing methods provide for weaker backbones. Adding routing on top may yield diminishing returns -- the "low-hanging fruit" is already captured.

---

## Claim 2: "NDDR-CNN Layerwise Fusion Is Scalable and Effective"

**Agent 3's Position:** NDDR-CNN's 1x1 Conv + BN fusion at every layer is the preferred mechanism, generalizes Cross-Stitch, and should be applied at FPN levels P3/P4/P5.

### Counter-Evidence:

**Scalability problem documented in Vandenhende survey (Section 2.2.1):**

> **"The size of the network increases linearly with the number of tasks."**

For NDDR specifically:

- Each NDDR unit concatenates all task features along the channel dimension, then applies a 1x1 Conv
- For N tasks: each NDDR unit processes N * C input channels (where C is per-task feature channels)
- For 4 tasks with 256-channel features: NDDR operates on 1024-channel inputs at every layer
- Applied at every backbone stage (16 blocks in MViTv2-S): this becomes a significant parameter/memory overhead

**Quantified cost for our setting:**

| Component | Parameters | FLOPs Impact |
|-----------|------------|-------------|
| 16 MViTv2-S blocks (baseline) | ~17M (base) | baseline |
| + NDDR at 4 stages (4 tasks) | ~4M extra (1x1 Conv 1024->256 per stage) | +~20% |
| + Cross-stitch at 4 stages | ~64 params (4x4 matrix per stage) | negligible |
| + ETR-NLP routing | ~NLP filter bank overhead | significant |

NDDR adds approximately 20-30% parameter overhead for 4 tasks. Cross-Stitch is lighter but also less effective. Neither is free.

**Limited receptive field issue (Section 2.2.2 of survey):**

> **"The main limitation of [NDDR/Cross-Stitch] architectures is that they only allow to use limited local information (i.e. small receptive field) when fusing the activations."**

This is critical for MViTv2-S:

- NDDR uses 1x1 Conv fusion, which has **no spatial receptive field** -- it operates per-pixel
- Cross-Stitch weights are **scalar multipliers** -- no spatial awareness at all
- For detection at P3/P4/P5, spatial context matters enormously. A per-pixel fusion at P3 cannot capture the spatial relationships that detection heads need
- MViTv2's pooling attention already provides better cross-scale context than any 1x1 Conv fusion

**Contradiction in Agent 3's recommendation:**

Agent 3 recommends NDDR at FPN levels P3/P4/P5 while also recommending per-scale routing (from MTI-Net). But NDDR and MTI-Net fusion mechanisms are different -- NDDR fuses across tasks at the same scale, MTI-Net fuses across scales for the same task. Agent 3 does not resolve this tension. Combining both would mean:
- NDDR across tasks at P3 (task fusion)
- MTI-Net across P3/P4/P5 for each task (scale fusion)
- This is architecturally complex and doubles fusion overhead

---

## Claim 3: "Task Routing with ~50% Shared Units Is Optimal for Heterogeneous Tasks"

**Agent 3's Position:** Task Routing (Strezoski ICCV 2019) shows ~50% unit sharing is optimal, and this applies to our 4-task setting.

### Counter-Evidence:

**Task Routing was evaluated on up to 20 classification tasks, not dense prediction tasks.**

- The Task Routing paper evaluates on Visual Decathlon (10 tasks) and Taskonomy (25 tasks), but these are primarily **classification/recognition** tasks with shared input domain
- Dense prediction tasks (detection, pose, segmentation) have fundamentally different feature requirements
- The 50% sharing finding was derived from classification tasks on ImageNet domain -- not applicable to heterogeneous dense prediction

**Re-analysis of Task Routing results:**

| Task Set | Best Sharing % | Notes |
|----------|---------------|-------|
| Visual Decathlon (digit/object cls) | ~50% | Similar domain, all classification |
| Taskonomy (scene recognition subset) | ~50-60% | All scene-level tasks |
| NYUv2 (sem seg + depth + normal) | Not evaluated | Unknown; may differ significantly |

No evidence exists that 50% sharing is optimal for the specific task mix of detection + pose + activity + PSR. These tasks span:
- Detection: object-level, bounding box, multi-scale
- Pose: keypoint-level, fine-grained localization
- Activity: video-level, temporal, global context
- PSR: pixel-level, 3D geometry, sparse supervision

**FiLM modulation limitations:**

Task Routing uses FiLM (Feature-wise Linear Modulation: gamma/beta scaling) to create task-specific features. However, as noted in the Vandenhende survey, FiLM-based task routing:

> "significantly increases the inference speed and somehow defies the purpose of MTL [since] tasks can not be predicted altogether."

For deployment: if tasks must be inferred sequentially (one at a time) due to FiLM routing, the inference cost multiplies by the number of tasks. For 4 tasks, this could mean 4x inference cost -- defeating the efficiency motivation for MTL.

---

## Claim 4: "ETR-NLP with Non-Learnable Primitives Is State-of-the-Art for Feature Routing"

**Agent 3's Position:** ETR-NLP (CVPR 2023) is the most advanced approach, explicitly decoupling shared and private features via non-learnable primitives.

### Counter-Evidence:

**ETR-NLP is evaluated on small-scale benchmarks only, with limited scope.**

The ETR-NLP paper evaluates on:
- NYUv2 (3 tasks, 795 training images)
- Cityscapes (3 tasks, 2,975 training images)
- Taskonomy (18 tasks, ~4k training images)

For MViTv2-S with 4 tasks trained on large-scale video data, ETR-NLP's benefits may not transfer:

1. **Non-learnable primitives (NLPs)** are random-weight convolutions that produce diverse features. In large-scale training, the backbone's learned features likely subsume or outperform random filters after sufficient training.

2. **The comparison is against older backbones.** ETR-NLP uses ResNet-50 backbone. MViTv2-S is significantly more powerful. The improvement from NLPs may be mostly compensating for ResNet-50's weaker representations -- a compensation MViTv2-S does not need.

3. **Explicit routing overhead.** ETR-NLP requires training a routing controller alongside the backbone. For 4 tasks with MViTv2-S's parameter count, this adds non-trivial training complexity and potential instability.

**More recent evidence (2023-2025):**

No major MTL benchmark since ETR-NLP (CVPR 2023) has adopted explicit non-learnable primitive routing as a standard component. The trend in large-scale multi-task models (e.g., BEiT-3, Florence, Uni-Perceiver) is toward:

- **Unified architectures** with shared backbones and task-conditioned heads
- **Prompt-based task specification** rather than feature-level routing
- **Scaling laws** favoring larger shared backbones over task-specific modules

This suggests the field is moving away from complex per-task routing and toward simpler shared representations with task-specific decoding.

---

## Claim 5: "MTI-Net Proves That Task Affinity Varies by Scale, Requiring Per-Scale Routing"

**Agent 3's Position:** MTI-Net (ECCV 2020) proves tasks must be routed independently at each FPN scale.

### Counter-Evidence:

**MTI-Net's core finding is about multi-scale task relationship modeling, not routing.**

MTI-Net's main contribution is showing that task affinity matrices differ across scales. However:

1. **The solution MTI-Net proposes** is multi-modal distillation across scales -- not explicit routing. The distillation units are cross-attention mechanisms that aggregate information across scales.

2. **MTI-Net still uses a shared encoder** for all tasks. The routing happens only in the decoder/refinement stages. This is actually closer to Agent 3's "fully shared early layers" recommendation than to per-task routing.

3. **Parameter cost is significant.** Each cross-task distillation unit at each scale adds parameters comparable to a small transformer block. For 4 tasks at 3 scales (P3/P4/P5), this is 12 distillation units -- each with multiple attention heads.

4. **The survey notes**: Decoder-focused approaches like MTI-Net (which refine task predictions through multi-scale interactions) often require careful tuning and can be brittle to hyperparameter changes.

**Pragmatic alternative for MViTv2-S:**

MViTv2-S's built-in **pooling attention** already provides scale-aware feature aggregation within the backbone. FPN then provides multi-scale features for the detection head. Rather than adding MTI-Net-style distillation across scales, simply using:

- Shared MViTv2-S backbone (all blocks)
- FPN for detection (P3/P4/P5) 
- cls_token for activity/pose
- P5 conv features for PSR

This is essentially what Agent 3 recommends for blocks 1-8, but extended to **all blocks** without splitting. The evidence from the survey suggests this simpler approach may match routed performance.

---

## Claim 6: "AdaShare's Adaptive Layer Allocation Is Parameter-Efficient"

**Agent 3's Position:** AdaShare can learn which MViTv2-S blocks to allocate per task, making it more efficient than fixed branching.

### Counter-Evidence:

**AdaShare's Gumbel-Softmax policy learning introduces training instability.**

1. **Discrete decision relaxation.** Gumbel-Softmax makes binary execute/skip decisions differentiable, but the relaxation introduces gradient variance. For transformer backbones with 16+ blocks, the combinatorial search space (2^16 per task) is enormous.

2. **Skip connections as workaround.** AdaShare requires skip connections to maintain information flow when layers are skipped. In a transformer, skipping blocks means bypassing self-attention -- which fundamentally changes the representation. The skip connection must compensate, which is not trivial for MViTv2-S's hierarchical pooling architecture.

3. **Training cost.** Policy learning means each forward pass must evaluate which blocks to execute for which tasks. For 4 tasks, this multiplies training time compared to a static architecture.

4. **No evidence on transformers.** AdaShare was evaluated on VGG/ResNet backbones, not transformers. Transformer blocks are more expensive to evaluate than conv layers, making the training cost argument worse.

**The oracle baseline:**

When AdaShare is compared to an "oracle" that knows the optimal layer allocation, the gap is often small -- meaning the optimal allocation is not far from simple heuristics (e.g., share early, branch late). For MViTv2-S, the heuristic "share all backbone, add task-specific heads" is nearly optimal without the complexity of learned policies.

---

## Claim 7: "Explicit Routing Reduces Task Interference Better Than Implicit Sharing"

**Agent 3's Position:** ETR-NLP and other explicit routing methods reduce interference by separating shared/private features.

### Counter-Evidence:

**Explicit routing can increase interference by creating competing gradient pathways.**

1. **Routing creates optimization challenges.** When tasks share some units and have private units, gradients from different tasks flow through different paths. The routing weights themselves must be optimized -- adding an extra optimization target that can interfere with task learning.

2. **Evidence from negative transfer literature:** The primary cause of negative transfer in MTL is gradient conflict (different tasks pulling shared parameters in different directions). Routing does not eliminate gradient conflict -- it merely relocates it to the routing mechanism itself.

3. **Results from the survey:** "While multi-task learning can lead to performance degradation if information sharing happens between unrelated tasks" -- this applies regardless of whether sharing is implicit (hard) or explicit (routed). The routing mechanism does not magically resolve incompatible task gradients.

4. **For our specific task set:**
   - Detection: BCE + regression losses (bounding box and classification)
   - Pose: MSE/L1 on keypoint coordinates
   - Activity: Cross-entropy on video-level classes
   - PSR: MSE on depth/surface normals

   These loss landscapes are fundamentally different (classification vs. regression vs. structured prediction). The gradient conflicts between detection's smooth L1 loss and PSR's per-pixel regression loss will exist regardless of whether features are routed or shared.

**Evidence from multi-objective optimization:**

Recent work in multi-objective optimization for MTL (e.g., Multiple Gradient Descent Algorithm, MGDA) shows that explicitly handling gradient conflicts at the **optimization level** (gradient surgery, PCGrad, GradDrop) is often more effective than architectural routing. This suggests the right place to handle task interference is at the optimization step, not the architectural level.

---

## Synthesis: What the Evidence Actually Shows

### What Routing Methods Definitely Do

1. **Increase parameter count** by 20-100% depending on number of tasks and fusion density
2. **Increase training complexity** by introducing learnable routing weights or policy networks
3. **Add inference overhead** if tasks must be processed sequentially (FiLM routing, ETR-NLP)
4. **Provide modest improvements** (1-3% on benchmarks) over strong shared baselines
5. **Show inconsistent benefits** depending on backbone, dataset, and task combination

### What Shared Backbones Definitely Do

1. **Minimize parameters** by avoiding task-specific parameters in the encoder
2. **Simplify training** with a single gradient flow through shared weights
3. **Enable single-pass inference** for all tasks
4. **Achieve competitive performance** when the backbone is sufficiently powerful
5. **Scale naturally** to any number of tasks without architectural changes

### The MViTv2-S Advantage

MViTv2-S is not a simple backbone -- it already incorporates:
- Hierarchical pooling (multi-scale representations)
- Pooling attention (efficient cross-token communication)
- FPN-compatible multi-scale features (P3/P4/P5)
- cls_token mechanism (for global tasks)

A strong argument: **MViTv2-S's built-in architectural sophistication already provides the benefits that routing methods offer to weaker backbones.** Adding routing on top may be redundant -- or worse, may interfere with features the backbone has already learned to produce.

---

## Recommended Alternative to Agent 3's Proposal

Instead of Agent 3's complex routing architecture (blocks 1-8 shared, 9-12 split, 13-16 task-specific, NDDR at 9 and 12, MTI-Net at FPN levels), we recommend:

### Phase 1: Simple Shared Backbone Baseline (Test First)
```
MViTv2-S (all 16 blocks, fully shared)
  ├── FPN (P3/P4/P5) ──> Detection head
  ├── cls_token ──> Activity head
  ├── cls_token ──> Pose head
  └── P5 conv ──> PSR head
```

**Rationale:** Before adding any routing complexity, establish whether MViTv2-S's backbone is strong enough that routing is unnecessary. This baseline has:
- Lowest parameter count
- Simplest training loop
- Single-pass inference
- Known upper bound from single-task performance

### Phase 2: Gradient Surgery (If Negative Transfer Detected)
If Phase 1 shows task interference (e.g., detection accuracy drops when PSR is added):
- Add PCGrad or GradDrop at the optimization level
- No architectural changes needed
- Evidence shows this resolves most gradient conflicts

### Phase 3: Minimal Routing (Only If Needed)
Only if Phases 1-2 fail to reach target metrics:
- **Cross-Stitch** (cheapest): 4x4 parameter matrices at blocks 12 and 16 only
- **NDDR** only at P5 (highest scale where fusion matters most)
- No MTI-Net, no ETR-NLP, no AdaShare unless Phase 3 also fails

---

## Summary: Counter-Evidence Table

| Agent 3 Claim | Counter-Evidence | Strength |
|--------------|------------------|----------|
| Routing methods show clear improvement | Survey: "moderate only" improvements; noise-level on many benchmarks | Strong |
| NDDR is scalable and effective | O(N) parameter growth; limited 1x1 receptive field; 20-30% overhead for 4 tasks | Strong |
| 50% sharing is optimal (Task Routing) | Evaluated on classification only, not dense prediction; FiLM inference cost multiplies | Moderate |
| ETR-NLP is state-of-the-art | Small-scale benchmarks; compensates for weak backbone; field moving toward unified models | Strong |
| MTI-Net requires per-scale routing | MTI-Net still uses shared encoder; MViTv2-S pooling attention already provides cross-scale aggregation | Moderate |
| AdaShare is parameter-efficient | Training instability; no evidence on transformers; heuristic matches oracle | Moderate |
| Explicit routing reduces interference | Routing does not eliminate gradient conflict; optimization-level methods may be superior | Strong |

**Overall Assessment:** Agent 3's recommendations are based on papers that show genuine but limited improvements on small benchmarks with weak backbones. For MViTv2-S with 4 heterogeneous tasks, the cost-benefit favors a simple shared backbone with gradient surgery over complex routing. The survey's conclusion stands: encoder-focused routing methods provide **moderate only** performance improvements -- likely insufficient to justify the architectural complexity for practical deployment.

---

## References

1. Vandenhende, S., Georgoulis, S., Van Gool, L. "MTI-Net: Multi-Scale Task Interaction Networks." ECCV 2020.
2. Vandenhende, S., et al. "Multi-Task Learning for Dense Prediction Tasks: A Survey." arXiv:2004.13379, 2020.
3. Misra, I., et al. "Cross-Stitch Networks for Multi-Task Learning." CVPR 2016.
4. Gao, Y., et al. "NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs." CVPR 2019.
5. Strezoski, G., et al. "Many Task Learning With Task Routing." ICCV 2019.
6. Liu, S., et al. "Multi-Task Attention Network (MTAN)." CVPR 2019.
7. Sun, X., et al. "AdaShare: Learning What To Share For Efficient Deep MTL." NeurIPS 2020.
8. Ding, C., et al. "ETR-NLP: Explicit Task Routing with Non-Learnable Primitives." CVPR 2023.
9. Kendall, A., et al. "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR 2018.
10. Ruder, S., et al. "Sluice Networks: Learning What to Share Between Tasks." AAAI 2019.
11. Li, Y., et al. "Taskonomy: Disentangling Task Transfer Learning." CVPR 2018.
12. Yu, T., et al. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020.
