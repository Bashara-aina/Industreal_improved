# 184 — Training & Data Deep Dive: Loss, Optimizer, Augmentation, Curriculum, Pretraining

**Date:** 2026-07-09
**Companion to:** 182 (strategy), 183 (architecture)
**Scope:** Everything outside the architecture itself — loss functions, optimizers, data augmentation, training schedules, pretraining strategy, distillation. The hidden 80% of any ML project's success.
**Purpose:** Concrete recipes + compute budgets for each training strategy in file 182.

---

## 0. The Training Problem in One Sentence

Even with the right backbone (file 183) and right heads, **the loss function determines whether the model converges at all**. The current setup uses F.cross_entropy + Focal(α=0.5) + BCE + cosine — a vanilla recipe that works for ImageNet but is suboptimal for 75-class long-tail assembly activity, sparse detection, and base-rate PSR. This file covers the upgrade menu.

---

## 1. Loss Function Upgrades per Task

### 1.1 Detection: YOLOv8 Loss (`v8DetectionLoss`)

**Current loss (post-Path-D):** focal(α=0.5, γ=2.0) + CIoU + DFL on 3×3 positive cells.

**Why it's insufficient:** center-cell / 3×3-patch assignment produces ~9 positive cells per GT. YOLOv8 uses `TaskAlignedAssigner` (TAL) which produces **10-50 cells per GT** dynamically based on alignment score `s = cls_score^α × box_iou^β`. The result is a 10× denser positive signal per GT, which is what allows YOLOv8 to converge to high mAP from a randomly initialized head.

**The full YOLOv8 detection loss:**
```python
# Simplified pseudocode of v8DetectionLoss
class V8DetectionLoss:
    def __init__(self, nc=24, reg_max=16, tal_topk=10):
        self.nc = nc
        self.reg_max = reg_max
        self.tal_topk = tal_topk  # 10 cells per GT
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dfl = DFLoss(reg_max)  # distribution focal
        
    def __call__(self, preds, targets):
        # preds: list of (B, h, w, 4*reg_max + nc) per FPN level
        # targets: (B, max_n, 5) (xyxy + cls)
        
        # 1. TaskAlignedAssigner — for each GT, pick top-K cells by alignment
        align_scores = cls_score ** alpha * box_iou ** beta  # alpha=1.0, beta=6.0 default
        topk_metrics, topk_idxs = align_scores.topk(self.tal_topk, dim=-1)
        
        # 2. Focal + BCE classification loss on assigned cells
        cls_loss = self.bce(cls_logits, target_cls)
        cls_loss *= focal_weight  # (1 - pt)^2
        
        # 3. CIoU + DFL bbox loss on assigned cells
        bbox_loss = (1 - ciou(pred_boxes, target_boxes)).mean()
        dfl_loss = self.dfl(pred_distr, target_distr).mean()
        
        return cls_loss + bbox_loss * 7.5 + dfl_loss * 1.5
```

**Key hyperparameters:** `tal_topk=10`, `tal_alpha=1.0` (classification weight in alignment), `tal_beta=6.0` (box IoU weight), `bbox_loss_gain=7.5`, `dfl_loss_gain=1.5`. These are YOLOv8 defaults and known-good.

**Why this works:** TAL is provably dense (every GT has 10 cells), adaptive (the assigned cells are the ones the model thinks are best aligned), and consistent across training (no hand-crafted assignment).

**Expected mAP@0.5 with YOLOv8 loss + MViTv2-S:** 0.65-0.85 (vs current 0). **Likely meets 80% bar.**

**Implementation:** Ultralytics YOLOv8 source code is BSD-3 licensed. Either copy verbatim (~600 lines) or `pip install ultralytics` and call its loss function. ~3-4 days of engineering.

**Recommendation:** **Adopt YOLOv8 head verbatim.** The cost is real but the alternative is failing to meet the bar.

### 1.2 Activity: ArcFace (additive angular margin)

**Current loss:** F.cross_entropy with class weights (sqrt-tamed, max 11.7) + label smoothing 0.05.

**Why it's insufficient:** CE with class weights treats all classes as equally "different." For fine-grained activity recognition where assembly states are visually similar (e.g., "10000000000" vs "10001000000"), the model needs to learn **angular separation** between class embeddings, not just cross-entropy on logits.

**ArcFace loss:**
```python
class ArcFaceHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=75, s=30.0, m=0.30):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s  # scale
        self.m = m  # angular margin (radians)
    
    def forward(self, features, targets):
        # features: [B, in_dim] (already pooled over time)
        # targets: [B]
        x = F.normalize(features, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        cos_theta = (x @ w.t()).clamp(-1 + 1e-7, 1 - 1e-7)  # [B, num_classes]
        
        # Add angular margin to the target class
        theta = torch.acos(cos_theta)
        target_theta = theta[torch.arange(len(targets)), targets]
        target_cos = torch.cos(target_theta + self.m)
        
        # One-hot encoding
        one_hot = F.one_hot(targets, num_classes=self.weight.size(0)).float()
        output = cos_theta * (1 - one_hot) + target_cos * one_hot
        output *= self.s
        
        return F.cross_entropy(output, targets)
```

**Key hyperparameters:** `s=30.0` (scale, standard), `m=0.30` (margin in radians). For very fine-grained 75-class: try `m=0.50`.

**Why this works:** ArcFace forces the model to learn class embeddings that are angularly separated by at least `m` radians. For fine-grained classes, this is a stronger signal than CE's softmax.

**Expected top-1 with ArcFace + 2-layer MLP + MViTv2-S:** 0.35-0.50 (vs current 0.008). **Likely meets bar with a stronger backbone; borderline with MViTv2-S alone.**

**Implementation:** ~150 lines. 1 day of engineering.

**Recommendation:** **Adopt ArcFace** if activity is the rate-limiting task.

### 1.3 PSR: Focal BCE (γ=2.0)

**Current loss:** BCE per component with inverse-prevalence weights.

**Why it's insufficient:** BCE on sparse positive labels is dominated by negatives. The current inverse-prevalence weights help, but at the cost of instabilities (some components have weights up to 137× others). Focal loss with γ=2.0 down-weights easy negatives automatically without explicit prevalence calculation.

**Focal BCE loss:**
```python
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        # logits: [B, T, 11]
        # targets: [B, T, 11] in {0, 1}
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()
```

**Why this works:** Focal γ=2.0 reduces loss contribution from confidently-classified cells (whether true or false) by ~100×. The model focuses on the 5-10% of cells that are uncertain, which are exactly the transition events we care about.

**Expected F1 with Focal BCE + block-3 features + 4-layer transformer:** 0.65-0.85 (vs current 0.0). **Likely meets bar with feature upgrade.**

**Implementation:** 30 lines. Half day.

**Recommendation:** **Adopt Focal BCE** for PSR. Cheap upgrade.

### 1.4 Pose: keep current (it works)

Cosine loss on already-good features. No upgrade needed. If we want to push beyond 4° MAE, we could add geodesic loss on rotation matrices (3× extra compute, 3-5 days engineering).

### 1.5 MTL loss: GradNorm (alternative to Kendall)

**Current loss:** Kendall uncertainty weighting + per-task caps + EMA normalization (Path-D).

**Why consider alternatives:** Path-D fixes Kendall's "weight = 1/(2·loss)" pathology but doesn't address the deeper question of *what* MTL balancing principle we want.

**GradNorm (Chen et al. 2018):** balances tasks by their **gradient norms**, not loss values. Each task's loss is weighted by a learnable scalar `w_i`, and `w_i` is updated so that the gradient norm of task `i` matches a target (the average gradient norm, or a moving average).

**Implementation:** ~100 lines. The key change: replace Kendall's `exp(-lv) * loss + lv/2` with a separate `weight_i * loss_i` where `weight_i` is updated by a meta-loss on gradient norms.

**Pros:** principled; doesn't degenerate to inverse-loss scaling.
**Cons:** requires 2nd-order gradients (slow); complex to implement correctly.

**Recommendation:** **Keep Path-D's Kendall-with-caps+EMA.** It's a simpler fix and is known to work for this codebase. GradNorm is a fallback if Path-D underperforms.

---

## 2. Optimizer Choices

### 2.1 Current: AdamW with cosine annealing

The current optimizer is AdamW with `lr_backbone=1e-4`, `lr_head=1e-3`, `lr_log_var=1e-3`, weight_decay=0.05, and a cosine annealing LR schedule over 100 epochs.

### 2.2 Upgrades worth considering

**Lion optimizer (Chen et al. 2023):**
- Sign-based momentum, ~3× lower memory than Adam.
- Better convergence on vision transformers.
- **Try:** `lr=3e-5` (Lion likes lower LR), `weight_decay=0.1`.

**Adafactor:**
- Sub-quadratic memory, good for large models.
- Useful when training InternVideo2-L (~304M params).
- Default lr `1e-3`.

**Schedule-free AdamW (Defazio et al. 2024):**
- No LR schedule needed; converges as well as cosine.
- Useful if we're not sure how many epochs we'll train.

**Recommendation:**
- **For Strat-1 (MViTv2-S):** keep AdamW + cosine. Already working.
- **For Strat-2 (InternVideo2-L):** try **Lion** with `lr=3e-5`, `weight_decay=0.1`. Lion is faster to converge on large models.
- **For Strat-4 (sequential pretrain):** use AdamW + cosine for the 4 pretraining runs, then AdamW + cosine for the MTL finetune.

### 2.3 LR schedule

**Current:** CosineAnnealingLR over 100 epochs from `lr_max` to ~0.

**Upgrade — cosine warmup + restart:**
- 5-epoch linear warmup from 1e-7 to `lr_max`.
- Cosine annealing over remaining epochs.
- One restart at epoch 50 (cosine annealing to 1e-6, then back to lr_max over 1 epoch).

**Why restart:** the model often gets stuck in a local minimum early. A restart gives it a chance to escape.

**Implementation:** `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)`.

### 2.4 Warmup specifically

Critical for training with frozen backbones (Strat-2). When backbone is frozen, the head + adapter gradients are large at step 0 and need warmup to avoid the head's initial large updates destabilizing the adapter.

**Recommended:** 10-epoch linear warmup for Strat-2. For Strat-1/4, 5-epoch warmup.

---

## 3. Data Augmentation

The current setup does not use much augmentation. Augmentation is one of the highest-leverage things we can change.

### 3.1 Standard video augmentation (recommended)

```python
class VideoAugment:
    def __init__(self, img_size=224):
        # Random resized crop
        self.rrc = RandomResizedCrop(size=img_size, scale=(0.7, 1.0))
        # Color jitter
        self.color = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        # Random horizontal flip (safe for assembly — assembly is symmetric)
        self.flip = RandomHorizontalFlip(p=0.5)
        # Temporal jittering: random frame drop + reorder (assembly-specific)
        self.temporal_jitter = TemporalJitter(max_drop=4, p=0.5)
        # Gaussian noise
        self.noise = GaussianNoise(std=0.02, p=0.3)
        # RandAugment (auto-select N=2 ops from M=10)
        self.randaug = RandAugment(num_ops=2, magnitude=10)
    
    def __call__(self, frames):
        # frames: [T, 3, H, W]
        # Apply spatial augmentations consistently across all frames in the clip
        params = self._sample_spatial_params()
        out = []
        for f in frames:
            f = self.rrc(f, params)
            f = self.flip(f, params)
            f = self.color(f)
            out.append(f)
        out = torch.stack(out)  # [T, 3, H, W]
        out = self.temporal_jitter(out)
        out = self.noise(out)
        return out
```

**Per-task augmentation strategy:**
- Detection: spatial-only augmentation (no temporal jittering — temporal consistency matters for bbox).
- Activity: aggressive spatial + temporal jittering. MixUp across videos (label mixing).
- PSR: minimal spatial augmentation. **No temporal jittering** — the temporal order is the supervision signal.
- Pose: spatial augmentation only.

### 3.2 MixUp / CutMix

MixUp and CutMix are powerful regularizers that interpolate samples. For multi-task, they require careful handling:

```python
class MultiTaskMixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, targets):
        # images: [B, T, 3, H, W]
        # targets: dict of per-task labels
        B = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        perm = torch.randperm(B, device=images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[perm]
        
        # For each task, mix labels appropriately:
        # - Detection: paste boxes from both images (clip boxes by image boundary)
        # - Activity: mix one-hot labels
        # - PSR: average targets (both sequences are valid)
        # - Pose: average 6D vectors (rotation blending is non-trivial — use SVD)
        mixed_targets = {}
        mixed_targets['activity'] = (lam * targets['activity_onehot'][perm] + 
                                     (1-lam) * targets['activity_onehot'])
        # ... etc
        
        return mixed_images, mixed_targets
```

**Per-task MixUp recommendation:**
- Detection: **paste boxes from both images** (don't blend pixel values). YOLOv8 has its own MixUp implementation.
- Activity: blend labels via `lam * onehot + (1-lam) * perm_onehot`.
- PSR: average targets across the time dimension.
- Pose: use **SVD-based rotation averaging** (convert 6D to rotation matrix, blend, project back to 6D). Without SVD, simply average 6D vectors (lossy but functional).

**Cost:** 2-3 days to implement correctly across all 4 tasks. Use Ultralytics's MixUp as a reference.

### 3.3 Domain-specific augmentation

For assembly videos specifically:
- **Frame masking:** randomly mask 1-3 frames in the 16-frame clip (drop them, replace with zeros, or replace with adjacent frames). Forces the model to interpolate.
- **Component swap augmentation:** for activity detection, swap two adjacent component states in 30% of clips. Tests the model's understanding of state sequences.
- **Speed change:** re-time the clip to 8 or 24 frames (downsample/upsample). Teaches temporal invariance.

---

## 4. Training Schedule (Epochs, Batches, Eval)

### 4.1 Current schedule
- 100 epochs, batch size 2, grad-accum 2 → effective batch 4 (now actually working post-Path-D).
- 4000 batches per epoch (capped from full 39,195 → only sees ~10% of data per epoch).
- Eval every 5 epochs on val split.
- Cosine annealing over 100 epochs.

### 4.2 The data cap problem

**4000 batches × 2 = 8000 samples/epoch.** Total training set is 78,391 windows = ~16× the per-epoch coverage. **At 100 epochs, we see each window ~6 times** (theoretically — in practice, weighted random sampling means some windows are seen 20+ times, others 0). The 4000-batch cap is a **severe bottleneck**.

**Two options:**
1. **Increase batch cap** to 8000 or 16000 (2× or 4× current). Doubles wall-clock per epoch but sees more data per epoch.
2. **Use full epoch** (39,195 batches). 10× wall-clock per epoch but converges faster per epoch.

**Recommendation:** **Increase to 8000 batches per epoch** (24 min/epoch instead of 22 min). Same effective epochs, but sees 2× more data. ~40 hours total. Within budget.

### 4.3 Eval frequency

**Current:** every 5 epochs. Each eval takes ~10-15 minutes on val (37796 windows).

**Recommendation:** Keep eval every 5 epochs. Don't reduce to 1 (too expensive) or increase to 10 (misses early signal).

### 4.4 Checkpointing

**Current:** saves `best.pt` (best activity top-1), `epoch_N.pt` (every 10 epochs), `latest.pt`.

**Recommendation:** Add `best_det.pt`, `best_psr.pt`, `best_pose.pt` so each task has its best checkpoint. Cost: 4× checkpoint storage (~2 GB).

---

## 5. Pretraining Strategy (Strat-4)

### 5.1 Per-task pretraining

For each task, train the backbone + task-specific head on that task alone (no MTL), using:
- AdamW, `lr=1e-4`, weight_decay=0.05
- 30 epochs, cosine annealing
- Path-D-style per-task log_var caps (with single task, this is unnecessary but harmless)
- Heavy augmentation (MixUp + CutMix + RandAug)
- YOLOv8 loss for detection, ArcFace for activity, Focal BCE for PSR, cosine for pose

Each pretraining run:
- Detection: ~2 GPU-days (MViTv2-S + YOLOv8 head)
- Activity: ~1 GPU-day (MViTv2-S + 2-layer MLP + ArcFace)
- PSR: ~2 GPU-days (MViTv2-S + 4-layer transformer + Focal BCE + block-3 features)
- Pose: ~0.5 GPU-days (already converges fast)

**Total: ~5-6 GPU-days.**

### 5.2 Model soup / task arithmetic

After the 4 pretraining runs finish, average the backbone weights:

```python
backbone_state_dicts = [
    torch.load("det_best.pt")["model_state_dict"]["feature_pyramid.backbone"],
    torch.load("act_best.pt")["model_state_dict"]["feature_pyramid.backbone"],
    torch.load("psr_best.pt")["model_state_dict"]["feature_pyramid.backbone"],
    torch.load("pose_best.pt")["model_state_dict"]["feature_pyramid.backbone"],
]

# Simple uniform average
avg_state_dict = {}
for key in backbone_state_dicts[0].keys():
    stacked = torch.stack([sd[key].float() for sd in backbone_state_dicts])
    avg_state_dict[key] = stacked.mean(dim=0)

# Save averaged backbone
torch.save(avg_state_dict, "soup_backbone.pt")
```

**Why this works (Wortsman et al. 2022, Ilharco et al. 2022 "task arithmetic"):** When models are trained from the same initialization on related tasks, their weights live in a connected basin of the loss landscape. Averaging them produces a model that's "between" the tasks — a strong starting point for joint MTL training.

### 5.3 MTL finetuning

Initialize the MTL model with the soup'd backbone + each task's trained head. Then finetune end-to-end with Path-D fixes:
- All 4 heads train jointly.
- 30 epochs (less than 100 because we start from a strong initialization).
- Path-D fixes (per-task caps + EMA + grad accum).
- Lower LR (5e-5 backbone, 5e-4 heads) to avoid catastrophic forgetting.

**Expected outcome:** MTL reaches 90-95% of single-task ceiling per head. **Likely meets 80% bar.**

---

## 6. Distillation (Strat-3)

### 6.1 The setup

Train 4 single-task teachers (same as Strat-4 pretraining, but possibly with stronger architectures). Then train an MTL student to mimic them.

### 6.2 Distillation loss

```python
class MultiTaskDistillationLoss(nn.Module):
    def __init__(self, teachers, alpha=0.5, temperature=2.0):
        self.teachers = teachers  # list of 4 frozen single-task models
        self.alpha = alpha  # balance between hard target and soft target
        self.T = temperature  # softmax temperature
    
    def __call__(self, student_outputs, teacher_outputs, hard_targets):
        # Distillation loss: KL divergence between student and teacher logits
        soft_loss = 0
        for task in ['activity', 'detection', 'psr', 'pose']:
            soft_loss += F.kl_div(
                F.log_softmax(student_outputs[task] / self.T, dim=-1),
                F.softmax(teacher_outputs[task] / self.T, dim=-1),
                reduction='batchmean'
            ) * self.T ** 2
        
        # Hard target loss (the usual task-specific loss)
        hard_loss = compute_task_loss(student_outputs, hard_targets)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

**Key hyperparameters:** `alpha=0.5` (50% soft, 50% hard), `temperature=2.0` (standard for KL distillation).

**Why this works:** the teacher's predictions carry more information than the one-hot labels (e.g., the teacher's distribution over classes reveals which classes are confusable). The student learns this richer signal.

**Expected outcome:** student reaches 85-95% of teacher per task. **Likely meets 80% bar.**

### 6.3 Implementation

~200 lines. 2 days of engineering. Well-understood recipe.

---

## 7. Compute Budget Per Strategy

| Strategy | Compute (GPU-days) | Wall-clock (1 GPU) | Notes |
|----------|-------------------|---------------------|-------|
| **Strat-1 incremental** | 4-6 | 4-6 days | Single run, simpler recipe |
| **Strat-2 frozen foundation** | 5-10 | 5-10 days | Single run with large backbone |
| **Strat-3 distillation** | 5-7 | 5-7 days | 4 ST + 1 distill |
| **Strat-4 sequential + finetune** | 7-10 | 7-10 days | 4 ST + 1 finetune |
| **4 single-task baselines** (always) | 4-6 | 4-6 days | Run in parallel with main path |

**Realistic total:** **8-15 GPU-days** if running all relevant strategies. At 1 primary GPU, that's **2-4 weeks wall-clock**.

If we have 2 GPUs available (the user mentioned 2), we can:
- Run Strat-1 + 4 ST baselines in parallel (5-6 GPU-days wall-clock).
- Then run Strat-2 or Strat-4 as the headline (5-10 more days).
- **Total: 2-3 weeks.**

---

## 8. Monitoring & Debugging

### 8.1 What to log per epoch

- Per-task losses (detailed breakdowns — see Opus 181's PSR per-component table).
- Per-task log_var (to verify Kendall caps are binding).
- Per-task EMA loss (to verify EMA is tracking, not exploding).
- Per-task grad norm (pre-PCGrad, post-PCGrad).
- Per-task backbone gradient contribution (% of total grad magnitude).
- Activity: top-k class predictions (to detect class collapse).
- Detection: distribution of predicted class scores per FPN level.
- PSR: per-component BCE breakdown.
- Pose: forward/up MAE separately.

### 8.2 Red flags to watch for

- **Activity class collapse:** all samples predict the same class → activity loss explodes.
- **Detection focal loss explosion:** if any positive cell has very high loss, check for label noise.
- **PSR loss flat at base-rate:** features still don't carry signal → check feature source.
- **Pose MAE > 30°:** the backbone is over-shifting; check backbone LR (lower it).
- **Gradient norm > 100:** explosion; reduce LR or increase gradient clipping.
- **EMA diverging:** loss is exploding; check NaN guards.

### 8.3 Quick health check (1-hour diagnostic)

For any new architecture change, run this 1-hour diagnostic before committing to a full training run:

1. **Forward pass on val:** verify output shapes match expected per head.
2. **Backward pass on 1 batch:** verify grad norms are finite and bounded (1-100).
3. **EMA convergence test:** run 100 batches and verify EMA converges within 10% of true mean.
4. **PCGrad conflict rate:** count % of task pairs with cosine < 0. Should be 10-30%; > 50% means high task conflict.
5. **Loss convergence test:** run 500 batches and verify loss decreases monotonically (with EMA smoothing).

If any of these fail, don't run a full training.

---

## 9. The Decision: What to Try First

Given compute and time constraints, here's the recommended trial sequence:

### Day 1: Smoke tests (1 GPU-day)

- Test InternVideo2-L backbone loading + 1 single-task activity epoch (with frozen backbone + 2-layer MLP). Report top-1.
- Test EVA-02-L the same way.
- Test YOLOv8 detection head on our existing MViTv2-S + best.pt. Run 1 epoch, report mAP.
- Test block-3 PSR features with current PSR head. Run 1 epoch, report F1.

These are 1-GPU-day experiments. Whichever wins informs the next steps.

### Day 2-3: Architecture decisions

Based on Day 1 results:
- **If InternVideo2 activity top-1 > 0.40** → Strat-2 with InternVideo2.
- **If YOLOv8 detection mAP > 0.40** → adopt YOLOv8 head (use in any Strat).
- **If block-3 PSR F1 > 0.30** → adopt block-3 features (use in any Strat).

### Day 4-7: Headline run

Launch the chosen Strat's headline run. Monitor per-epoch.

### Day 8-10: Single-task baselines

Run 4 single-task baselines (in parallel if 2 GPUs available).

### Day 11-14: Analysis & paper writeup

Compute SOTA ratios, write up results.

**Total: 2 weeks** for the full program. Aggressive but feasible.

---

## 10. Open Training & Data Questions for Opus

See file 185 for the full 50. The training-side ones most worth flagging here:

- **T-1:** Does MixUp help long-tail activity classification (75 classes), or does it make the rare classes even rarer?
- **T-2:** Should we use EMA model weights (Polyak averaging) in addition to the EMA loss tracker?
- **T-3:** Is 8000 batches/epoch enough data coverage, or do we need the full 39,195?
- **T-4:** For Strat-4's model soup, do we average only the backbone weights, or also the head weights?
- **T-5:** Is Focal BCE (γ=2.0) better than BCE-with-inverse-prevalence for PSR?
- **T-6:** Does RandAugment work for video data, or do we need video-specific augmentation (e.g., temporal jitter)?
- **T-7:** Should we use mixed-precision (bf16) consistently, or revert to fp32 for the head modules?
- **T-8:** How important is gradient clipping norm? Current is 1.0; should we increase to 5.0?

---

*Companion to 182 (strategy), 183 (architecture), 185 (questions). Compute budgets are upper bounds; actual times depend on data-loading bottlenecks and GPU availability.*