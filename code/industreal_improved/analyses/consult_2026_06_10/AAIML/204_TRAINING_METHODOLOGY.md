# 204 — Training Methodology: Exact Hyperparameters, Schedule, and Loss Functions

**Date:** 2026-07-10
**Sources:** Research agents on training recipes, distillation, curriculum learning, long-tail classification, data augmentation

---

## 1. Complete Training Recipe

### 1.1 Phase 1: Independent Adapter Training (Epochs 1-5)

**Goal:** Each LoRA+FiLM adapter reaches near-ST performance without gradient interference.

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Backbone | Frozen (Kinetics-400 pretrained) | Preserves pretrained features |
| Trainable params | LoRA+FiLM adapters (4.2M) + heads (~15M) | 19.2M total trainable |
| Batch size | 2 × grad-accum-2 = effective 4 | VRAM-limited 12GB |
| Optimizer | AdamW | Standard for transformer fine-tuning |
| LR (adapters) | 1e-3 | Fast adaptation needed |
| LR (heads) | 1e-3 | Fresh heads need higher LR |
| Weight decay | 0.05 | Standard for AdamW on transformers |
| Scheduler | None for this phase | Constant LR during adapter warmup |
| Gradient clipping | 5.0 norm | Prevents adapter explosion |
| Mixed precision | BF16 autocast | 30% speed gain, stable on CUDA |
| Max batches/epoch | 4000 | ~17 min per epoch per task |
| Training pattern | Independent (NOT joint) | Each task trained separately |

**Training sequence:**
1. Train detection adapters (epochs 1-5) — 85 minutes
2. Train activity adapters (epochs 1-5) — 85 minutes
3. Train PSR adapters (epochs 1-5) — 85 minutes
4. Train pose adapters (epochs 1-5) — 85 minutes

**Total Phase 1 time:** ~6 hours (sequential on GPU 1). Could be parallelized across 2 GPUs.

### 1.2 Phase 2: Joint Multi-Task Fine-tuning (Epochs 6-50)

**Goal:** All adapters + backbone trained jointly. Cross-task transfer through shared backbone.

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Backbone | Unfrozen (all layers) | Enable cross-task transfer |
| Trainable params | All (60M) | Full joint optimization |
| Batch size | 2 × grad-accum-2 = effective 4 | Same as Phase 1 |
| Optimizer | AdamW | |
| LR (backbone) | 1e-5 | 100× lower than heads — protect pretrained weights |
| LR (adapters) | 1e-4 | 10× lower than Phase 1 — fine-tuning mode |
| LR (heads) | 1e-4 | Same as adapters |
| Weight decay | 0.05 | |
| Scheduler | Cosine annealing to 1e-7 over epochs 6-50 | Smooth decay |
| Gradient clipping | 2.0 norm | Lower for joint training stability |
| Mixed precision | BF16 autocast | |
| Max batches/epoch | 8000 | ~35 min per epoch |
| Eval every | 5 epochs | |
| Warmup | 500 linear warmup steps at start of Phase 2 | Stabilize backbone unfreezing |

**Gradient management: Nash-MTL (replaces PCGrad).**

```python
def nash_mtl_step(losses, shared_params):
    """Nash-MTL: game-theoretic gradient bargaining (Navon et al. ICML 2022)."""
    grads = {}
    for name, loss in losses.items():
        # Compute per-task gradient
        grad = torch.autograd.grad(loss, shared_params, retain_graph=True)
        grads[name] = [g.detach().clone() for g in grad]

    # Compute Nash bargaining solution
    n_tasks = len(losses)
    grad_matrix = torch.stack([torch.cat([g.flatten() for g in grads[t]]) for t in grads])
    # Gram matrix G_ij = <grad_i, grad_j>
    G = grad_matrix @ grad_matrix.T  # [n_tasks, n_tasks]
    # Nash weights: solve G·α = 1
    try:
        alpha = torch.linalg.solve(G, torch.ones(n_tasks, device=G.device))
        alpha = alpha / alpha.sum()  # normalize to sum=1
    except torch.linalg.LinAlgError:
        alpha = torch.ones(n_tasks, device=G.device) / n_tasks  # fallback to uniform

    # Weighted gradient combination
    combined_grad = sum(alpha[i] * torch.cat([g.flatten() for g in grads[list(losses.keys())[i]]])
                        for i in range(n_tasks))
    return combined_grad, alpha
```

**Why Nash-MTL over PCGrad:**
- PCGrad only handles pairwise conflicts (O(n²) comparisons). Nash-MTL solves the global bargaining problem (O(1)).
- Published MTL/ST improvement: +2-8% over PCGrad on standard benchmarks
- Nash-MTL has convergence guarantees that PCGrad lacks

### 1.3 Phase 3: Per-Head Fine-tuning (Epochs 51-70, Optional)

**Goal:** If any head lags significantly behind its ST baseline, run additional epochs with:
- That head's LR 10× higher than others
- Nash-MTL weight manually boosted for lagging head
- Cosine annealing to 0

This phase is conditional on eval results from Phase 2.

---

## 2. Knowledge Distillation (Optional Phase)

### 2.1 ST Teacher Training

Train 4 single-task specialists with 2× head capacity (deeper heads, higher LR, longer training). These don't need to be parameter-efficient — they're teachers, not the deployed model.

| Teacher | Architecture | Params | Training epochs |
|---------|-------------|--------|-----------------|
| Detection | BiFPN + GFLV2 head ×4 layers | ~5M head | 50 |
| Activity | Temporal pool + 4-layer MLP | ~8M head | 50 |
| PSR | 4-layer hierarchical T | ~8M head | 50 |
| Pose | 4-layer MLP + geodesic | ~1M head | 30 |

### 2.2 Distillation Loss

```python
def distillation_loss(student_outputs, teacher_outputs, student_targets, temperature=3.0):
    """Combined task loss + KL distillation from ST teachers."""
    # Task-specific losses (same as Phase 2)
    L_task = sum(task_losses)

    # KL distillation per head
    L_distill = 0.0
    for task in ["det", "act", "psr", "pose"]:
        s_logits = student_outputs[task] / temperature
        t_logits = teacher_outputs[task] / temperature
        L_distill += temperature**2 * F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction='batchmean'
        )

    # Feature-level distillation (FitNets-style)
    L_feature = 0.0
    for layer_name in ["block_8", "block_12", "block_16"]:
        s_feat = student_intermediate[layer_name]
        t_feat = teacher_intermediate[layer_name]
        L_feature += F.mse_loss(s_feat, t_feat)

    alpha = 0.5  # task loss weight
    beta = 0.4   # logit distillation weight
    gamma = 0.1  # feature distillation weight
    return alpha * L_task + beta * L_distill + gamma * L_feature
```

**Expected MTL/ST ratio with distillation: 93-97%** (from Agent a5df69694 research).

---

## 3. Data Augmentation Strategy

### 3.1 Detection-Specific Augmentation (from Agent ab7a1846)

```
Mosaic(probability=0.75)  — combine 4 clips into one, boosts small-object mAP
MixUp(alpha=0.2, beta=0.5)  — mix two clips, regularization
CopyPaste(probability=0.3)  — paste objects across clips, boosts rare classes
RandomAffine(scale=[0.5, 1.5], degrees=[-10, 10])  — handle depth/angle variation
HorizontalFlip(probability=0.5)  — assembly is not left-right invariant but some parts are symmetric
```

**Mosaic schedule:** Active epochs 1-40, disabled epochs 41-70. Mosaic in late training destabilizes fine-grained detection.

### 3.2 Activity Augmentation

```
MultiCrop: 1 global (224×224) + 2 local (160×160) per clip
Temporal jitter: randomly shift clip window ± 3 frames
Frame drop: randomly drop 1-2 frames per 16-frame clip (simulates occlusions)
```

### 3.3 PSR Augmentation

```
Random temporal crop: use 12-16 frames instead of always 16
Frame rate jitter: sample every 1-3 frames instead of always stride=1
```

### 3.4 Pose Augmentation

```
Rotation augment: rotate clip ±5° and adjust GT pose accordingly
No color augment: lighting changes don't affect head direction
```

---

## 4. Long-Tail Activity Strategy

Based on Agent ada79e17e research and our own logit-adjust:

**Strategy: Decoupled Training (Kang et al. ICLR 2020)**

```
Phase A (epochs 1-25): Instance-balanced sampling
  - Each clip sampled uniformly, regardless of class
  - Learns general features for all classes

Phase B (epochs 26-50): Class-balanced classifier retraining
  - Freeze backbone + temporal pool
  - Retrain only classifier (last Linear layer) with class-balanced sampling
  - Each class sampled equally often (oversampling rare classes)
  - This is the critical fix for long-tail: the features are good, the classifier is biased
```

**Combined with logit-adjust:**

```python
# Phase A: use class weights for balanced gradient
loss = F.cross_entropy(logits, targets, weight=class_weights)

# Phase B: freeze features, retrain classifier with balanced batches + logit-adjust
loss = F.cross_entropy(logits, targets)  # balanced batches, no weights needed
# Logit-adjust corrects prior at inference time
```

**Expected gain from decoupled training alone:** +5-10% top-1 on long-tail classes.

---

## 5. Complete Hyperparameter Table

| Phase | Epochs | Backbone | LR Backbone | LR Heads | LR Adapters | Scheduler | Grad Clip | Aug |
|-------|--------|----------|-------------|----------|-------------|-----------|-----------|-----|
| 1A: Det adapter | 1-5 | Frozen | — | 1e-3 | 1e-3 | None | 5.0 | Det |
| 1B: Act adapter | 1-5 | Frozen | — | 1e-3 | 1e-3 | None | 5.0 | Act |
| 1C: PSR adapter | 1-5 | Frozen | — | 1e-3 | 1e-3 | None | 5.0 | PSR |
| 1D: Pose adapter | 1-5 | Frozen | — | 1e-3 | 1e-3 | None | 5.0 | Pose |
| 2: Joint MTL | 6-50 | Unfrozen | 1e-5 | 1e-4 | 1e-4 | Cosine→1e-7 | 2.0 | All |
| 3: Per-head boost | 51-70 | Unfrozen | 1e-6 | 1e-4* | 1e-5 | Cosine→0 | 2.0 | All |

*Boosted head gets 4e-4, others stay at 1e-4.

---

## 6. Expected Training Time

| Phase | GPU hours (1× RTX 3060 12GB) |
|-------|------------------------------|
| Phase 1 (4 tasks independent) | ~6 hours |
| Phase 2 (joint MTL epochs 6-50) | ~26 hours (45 epochs × 35 min) |
| Phase 3 (optional boost) | ~12 hours |
| **Total** | **~32-44 hours** |

**Plus ST teacher training (for distillation):** ~20 hours per teacher × 4 = ~80 GPU-hours on GPU 2.

---

## 7. Evaluation Protocol

### 7.1 Metrics (unchanged from current eval)

| Task | Primary Metric | Secondary | Monitoring |
|------|---------------|-----------|------------|
| Detection | mAP@0.5 | mAP@0.5:0.95, presence accuracy | Per-class AP for rare classes |
| Activity | top-1 accuracy | top-5, per-class F1 | Logit-adjust coefficient |
| PSR | event_F1@±3 | POS, τ (seconds) | Per-component precision/recall |
| Pose | fwd MAE | up MAE, geodesic MAE | Bootstrap 95% CI |

### 7.2 Ablation Experiments

For the Kendall-collapse paper story (Opus 201's Figure 1):

1. **Full MTL with caps + Nash-MTL** (our proposed method)
2. **Full MTL with uncapped Kendall** (`--kendall-uncapped`)
3. **Full MTL with PCGrad instead of Nash-MTL**
4. **Single-task per head** (the 4 ST baselines)
5. **MTL without adapters** (no LoRA, no FiLM — backbone directly shared)

All five configurations run on the same data, same eval, same number of epochs. This gives the paper its methodological contribution.
