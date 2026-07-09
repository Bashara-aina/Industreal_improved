# 190 — Training Path & Hypothesis Validation: How to Prove MTL is Helping, Not Hurting

**Date:** 2026-07-09
**Companion to:** 187 (Opus 181+186 status), 188 (per-head upgrades), 189 (backbone + MTL topology)
**Purpose:** The complete training path from cold start to a paper-ready MTL model. Includes the hypothesis validation framework (how to *prove* MTL is helping across all heads, not just claiming it).
**Hypothesis to support:** "MTL with one shared backbone is more efficient, faster, and **at least as accurate as single-task specialists on each head, often beating SOTA**."

---

## 0. The Validation Problem in One Sentence

The hypothesis is **"MTL is helping across the board"** — not "MTL reaches X% of SOTA." To prove this, we need:
1. **Per-task single-task baselines** on the same backbone, same data, same hyperparams.
2. **MTL results on the same backbone + heads + data.**
3. **A clear comparison framework** showing MTL is at least competitive with single-task on each head, while being more parameter-efficient and faster.

The single-task baselines are non-negotiable. Without them, the paper's MTL claims are unprovable.

---

## 1. THE COMPLETE TRAINING PATH

### 1.1 Three-phase plan

**Phase 1 — Per-head architecture upgrade + smoke tests (3-5 days)**
- Implement YOLOv8 detection head (file 188 §1.3)
- Implement STORM-like PSR decoder (file 188 §3.3)
- Implement temporal attn pool activity head (file 188 §2.4)
- Run 1-epoch ablation on each head independently to verify it works
- Run 1-epoch ablation with foundation backbone (if chosen)

**Phase 2 — Per-task single-task pretraining (4-5 GPU-days total)**
- Train each task alone (4 separate runs)
- Each run uses the chosen backbone + head
- Each run produces a "specialist" model for that task
- These specialists are the **comparison baseline** for the paper

**Phase 3 — MTL finetune + model soup (5-7 GPU-days)**
- Initialize MTL model with averaged (soup'd) backbone from Phase 2
- Initialize each head with the corresponding Phase 2 specialist
- Finetune end-to-end at low LR (5e-5 backbone, 5e-4 heads)
- Path-D fixes (per-task log_var caps + EMA + grad accum)
- EMA model weights

**Total: 2-3 weeks wall-clock (single GPU), or 1-1.5 weeks with 2 GPUs in parallel.**

### 1.2 Compute budget per phase

| Phase | GPU-days | Wall-clock (1 GPU) | Wall-clock (2 GPUs) |
|-------|----------|---------------------|----------------------|
| Phase 1 (architecture + smoke) | 1-2 | 1-2 days | 1-2 days |
| Phase 2 (4 ST pretrains) | 4-5 | 4-5 days | 2-3 days |
| Phase 3 (MTL finetune) | 5-7 | 5-7 days | 5-7 days |
| Eval + analysis | 1-2 | 1-2 days | 1-2 days |
| **Total** | **11-16** | **11-16 days** | **9-14 days** |

This is doable for an AAIML submission with a 2-3 week timeline.

---

## 2. PHASE 1 — Per-Head Architecture Implementation + Smoke Tests

### 2.1 Smoke test plan (1-2 days total)

For each of the 4 heads, run a **1-epoch single-task smoke test** to verify the new architecture works:

```bash
# Detection smoke test
python scripts/train_mtl_mvit.py --plumbing --task det_only --epochs 1 \
    --output-dir /tmp/smoke/det

# Activity smoke test
python scripts/train_mtl_mvit.py --plumbing --task act_only --epochs 1 \
    --output-dir /tmp/smoke/act

# PSR smoke test
python scripts/train_mtl_mvit.py --plumbing --task psr_only --epochs 1 \
    --output-dir /tmp/smoke/psr

# Pose smoke test
python scripts/train_mtl_mvit.py --plumbing --task pose_only --epochs 1 \
    --output-dir /tmp/smoke/pose
```

**Decision criteria for each smoke test:**
- ✅ Forward pass produces expected output shape
- ✅ Loss decreases in the first 100 steps
- ✅ Backward pass succeeds (no NaN/Inf)
- ✅ mAP / top-1 / F1 / MAE improves vs initial random baseline

**If any head fails the smoke test:** debug before Phase 2.

### 2.2 Foundation backbone smoke test (1 day)

If using a frozen foundation model (Tier 2 plan, file 189 §3.2):

```bash
# Test DINOv2-L (or InternVideo2-L if license cleared)
python scripts/train_mtl_mvit.py --plumbing \
    --backbone dinov2_l_frozen --epochs 1 \
    --output-dir /tmp/smoke/dinov2
```

**Decision criteria:**
- ✅ Forward pass + LoRA adapter works
- ✅ Activity top-1 at ep1 > 0.10 (vs current 0.008)
- ✅ PSR loss at ep1 < 1.0 (vs current 1.30)

**If foundation backbone fails:** fall back to MViTv2-L (Tier 1 plan).

---

## 3. PHASE 2 — Per-Task Single-Task Pretraining

### 3.1 Why this is critical

Without single-task baselines, the paper's MTL claims are unprovable. We need:
- "ST-detection reaches 0.75 mAP, MTL reaches 0.70 mAP" → MTL cost is small
- "ST-activity reaches 0.55 top-1, MTL reaches 0.50 top-1" → MTL cost is small
- Etc.

**These are the comparison numbers that make the paper credible.**

### 3.2 ST detection training (1-2 GPU-days)

```python
# scripts/train_st_det.py
"""Single-task detection training.

Trains YOLOv8 head on top of (chosen backbone) using TAL + DFL + CIoU loss.
Target: reach near-SOTA detection (mAP@0.5 ≈ 0.65-0.85).
"""

import torch
import torch.nn as nn
from src.models.yolov8_mtl_det_head import YOLOv8DetectionModel
from src.losses.v8_detection_loss import V8DetectionLoss
from src.data.industreal_dataset import IndustRealDetDataset, collate_fn

# ... build dataset, model, loss ...
# ... train for 30 epochs ...
```

**Expected:** mAP@0.5 = 0.65-0.85 with foundation backbone; 0.45-0.70 with MViTv2-L.

**Cost:** 1-2 GPU-days.

### 3.3 ST activity training (1 GPU-day)

```python
# scripts/train_st_act.py
"""Single-task activity training.

Trains temporal attn pool + 2-layer MLP + ArcFace on top of (chosen backbone).
Target: top-1 ≈ 0.50-0.70.
"""

# ... build dataset, model, loss (ArcFace), optimizer ...
# ... train for 20-30 epochs ...
```

**Expected:** top-1 = 0.50-0.70 with foundation backbone; 0.25-0.45 with MViTv2-L.

**Cost:** 1 GPU-day.

### 3.4 ST PSR training (1-2 GPU-days)

```python
# scripts/train_st_psr.py
"""Single-task PSR training.

Trains STORM-like decoder on top of (chosen backbone).
Target: F1@±3 ≈ 0.60-0.80.
"""

# ... build dataset, model (STORM decoder), loss (Focal BCE) ...
# ... train for 30 epochs ...
```

**Expected:** F1@±3 = 0.60-0.80 with foundation backbone; 0.40-0.65 with MViTv2-L.

**Cost:** 1-2 GPU-days.

### 3.5 ST pose training (0.5 GPU-day, optional — pose is already done)

Pose is already training well in the current MTL run. **Skip Phase 2 for pose**, or run a quick ST-pose to get the comparison number.

**Cost:** 0.5 GPU-day.

### 3.6 What Phase 2 gives us

After Phase 2, we have:
- 3-4 single-task specialist models (one per head)
- Per-head ceiling numbers (what ST can reach on this backbone)
- The backbone weights for each task — basis for model soup

---

## 4. PHASE 3 — MTL Finetune from Soup Initialization

### 4.1 Why model soup

Wortsman et al. 2022 + Ilharco et al. 2022 (task arithmetic) showed that **averaging weights of fine-tuned models** (from the same init) often produces a stronger model than each individual. The averaged weights live in a "connected basin" of the loss landscape.

For our case: averaging 4 task-specific backbones should produce a backbone that's "good for all tasks" — a strong starting point for MTL finetuning.

### 4.2 Soup construction

```python
# scripts/build_soup.py
"""Build model soup by averaging backbone weights from Phase 2 specialists."""
import torch
import torch.nn as nn
from collections import OrderedDict

# Load 3-4 specialist backbones
specs = [
    torch.load("checkpoints/st_det/best.pt")["model_state_dict"]["backbone"],
    torch.load("checkpoints/st_act/best.pt")["model_state_dict"]["backbone"],
    torch.load("checkpoints/st_psr/best.pt")["model_state_dict"]["backbone"],
    # Skip pose (already healthy in MTL)
]

# Average
avg_state = OrderedDict()
for key in specs[0].keys():
    stacked = torch.stack([s[key].float() for s in specs])
    avg_state[key] = stacked.mean(dim=0).to(specs[0][key].dtype)

# Save averaged backbone
torch.save(avg_state, "checkpoints/soup_backbone.pt")
```

**Cost:** 5 minutes (just weight averaging).

### 4.3 MTL finetune from soup

```python
# scripts/train_mtl_finetune.py
"""MTL finetune from soup initialization.

Loads soup'd backbone + each Phase 2 head, then finetunes end-to-end
with Path-D fixes at low LR.
"""

# ... load soup backbone + each specialist head ...
# ... finetune for 30-50 epochs at lr_backbone=5e-5, lr_head=5e-4 ...
# ... use Path-D fixes (per-task caps + EMA + grad accum) ...
```

**Expected:** 0.65-0.85 detection, 0.45-0.65 activity, 0.55-0.80 PSR, 4-6° pose. **Likely meets 80% SOTA across heads.**

**Cost:** 5-7 GPU-days (or 3-4 days on 2 GPUs).

---

## 5. HYPOTHESIS VALIDATION FRAMEWORK

### 5.1 The hypothesis statement

> "MTL with one shared backbone achieves comparable or better per-task performance than single-task specialists, while being more parameter-efficient and faster (in training, not just inference)."

This is a **stronger claim** than Opus 181's L2+L3+method. It requires **MTL ≥ ST on at least 3/4 tasks** and **MTL ≪ ST in parameter count**.

### 5.2 What we need to measure

For each task, we need:
1. **ST performance** (Phase 2 results)
2. **MTL performance** (Phase 3 results)
3. **MTL/ST ratio** (a number < 1 means MTL is worse; = 1 means same; > 1 means MTL is better)
4. **Statistical significance** (need ≥3 seeds for confidence intervals)

### 5.3 The headline table (target)

| Head | ST (Phase 2) | MTL (Phase 3) | MTL/ST ratio | SOTA | MTL/SOTA |
|------|--------------|---------------|--------------|------|----------|
| Detection mAP@0.5 | 0.75 | 0.72 | 0.96 | 0.838 | 0.86 |
| Activity top-1 | 0.60 | 0.58 | 0.97 | 0.652 | 0.89 |
| PSR F1@±3 | 0.70 | 0.68 | 0.97 | 0.901 | 0.75 |
| Pose fwd MAE | 4.5° | 4.8° | 0.94 | No SOTA | n/a |

**MTL/ST ratio ≥ 0.9 on all 4 heads = hypothesis supported.**

### 5.4 What if MTL is worse than ST on some head?

This is the most likely outcome for one of the heads (probably PSR or activity). The paper then has two stories:

**Story A (positive transfer):** "MTL reaches 0.95× of ST ceiling across 3/4 heads, with 3× fewer parameters. This demonstrates MTL is highly parameter-efficient."

**Story B (mixed):** "MTL beats ST on 2/4 heads (positive transfer) and matches ST on 1/4 heads. PSR has 0.85× of ST due to [specific reason]. This reveals the per-task characteristics where MTL is/isn't beneficial."

Both are publishable. Story A is the goal; Story B is acceptable.

### 5.5 What if MTL is much worse than ST?

If MTL/ST < 0.7 on multiple heads:
- The MTL hypothesis fails
- Revert to single-task models (Strat-5 from file 182)
- Publish as "MTL pathology" paper (Opus 181's Option 3)

This is the worst-case outcome. **To avoid it, we need the head upgrades (file 188) + frozen foundation backbone (file 189) + proper training (Phase 2 + 3 here).**

---

## 6. EFFICIENCY CLAIMS

The "MTL is more efficient" claim needs concrete numbers:

### 6.1 Parameter efficiency

| Approach | Total Params | Per-task Params | Notes |
|----------|--------------|-----------------|-------|
| 4 ST specialists | 4 × ~100M = 400M | 100M each | Each specialist has its own backbone |
| MTL (current) | 43.5M | 10.9M each (backbone) + head | Shared backbone |
| MTL (frozen foundation + LoRA) | 304M (frozen) + 5M (LoRA) = 309M | 1.25M (LoRA) + head | Foundation is reusable |

For the frozen foundation case, **MTL is ~26% more parameter-efficient** than 4 ST specialists (309M vs 400M). With MViTv2-S, MTL is **9.2× more efficient** (43.5M vs 400M).

### 6.2 Training efficiency

For the **frozen foundation** case:
- Backbone is frozen; no gradients to compute
- Only LoRA + heads train: ~5-10M params
- Forward pass: 1× (one backbone) vs 4× (one per specialist)
- Per-epoch wall-clock: ~30% of ST total

For the **MViTv2-S shared** case:
- Backbone trains: 34.5M params
- Forward pass: 1× (one backbone) vs 4×
- Per-epoch wall-clock: ~25% of ST total

**MTL is ~3-4× faster in training wall-clock** (single GPU; less so with multi-GPU ST parallelism).

### 6.3 Inference efficiency

For inference (deployment):
- 1 forward pass for 4 tasks (vs 4 for ST)
- 1 model to load (vs 4)
- ~2-3× lower latency in batched inference
- ~2-3× lower memory footprint

**These are concrete, measurable claims that the paper can defend.**

---

## 7. WHAT IF WE DON'T HAVE TIME FOR ALL 3 PHASES?

### 7.1 Phase 1 only (3-5 days)

**Output:** New head architectures work individually. MTL run continues with new heads.

**What we get:** Improved MTL numbers (likely 50-65% of SOTA, not 80%).

**Paper story:** "MTL with carefully designed heads reaches X% of SOTA at 1/9th the parameter cost of 4 specialists."

**Publishable:** Yes, but as L2+L3+method, not "MTL beats SOTA."

### 7.2 Phase 1 + Phase 2 (8-10 days)

**Output:** Single-task baselines. Per-task ceiling numbers.

**What we get:** Honest comparison. "MTL reaches Y% of ST ceiling" with Y measured, not estimated.

**Paper story:** "MTL reaches 90% of single-task ceiling at 3× fewer parameters" — strong claim with concrete numbers.

**Publishable:** Yes, very defensible. Doesn't require beating SOTA.

### 7.3 Phase 1 + Phase 2 + Phase 3 (15-20 days)

**Output:** MTL model finetuned from soup. Hypothesis testable.

**What we get:** MTL with possible SOTA-comparable performance. Concrete hypothesis validation.

**Paper story:** "MTL reaches 80% of SOTA on 3/4 heads, at 1/9th the parameter cost" — strongest claim.

**Publishable:** Best-case outcome.

### 7.4 Time-constrained decision tree

```
Q: How much time do we have?
  ├─ < 1 week → Phase 1 only (head upgrades)
  ├─ 1-2 weeks → Phase 1 + Phase 2 (ST baselines)
  └─ 2-3 weeks → Phase 1 + 2 + 3 (full plan)
```

**For the current 3-week timeline, all 3 phases fit.**

---

## 8. SPECIFIC IMPLEMENTATION DETAILS

### 8.1 Single-task detection training script (skeleton)

```python
# scripts/train_st_det.py
"""Single-task detection on chosen backbone."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.backbones import get_backbone
from src.models.yolov8_head import YOLOv8Head
from src.losses.v8_detection_loss import V8DetectionLoss
from src.data.industreal_dataset import IndustRealDetDataset, collate_fn_det
from src.evaluation.evaluate import compute_det_metrics_extended

def main():
    # Build dataset
    train_ds = IndustRealDetDataset(split="train", img_size=(224, 224))
    val_ds = IndustRealDetDataset(split="val", img_size=(224, 224))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn_det)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn_det)

    # Build model: backbone + YOLOv8 head
    backbone, backbone_dim = get_backbone("dinov2_l_frozen")  # or "mvitv2_l"
    head = YOLOv8Head(nc=24, reg_max=16, ch=(256, 512, 1024))
    model = nn.Sequential(backbone, head)

    # Optimizer
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Loss
    criterion = V8DetectionLoss(nc=24, tal_topk=10)

    # Train
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            images, targets = batch
            features = backbone(images)
            preds = head(*features)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval
        if epoch % 5 == 0:
            mAP = evaluate_det(model, val_loader)
            print(f"Epoch {epoch}: mAP@0.5 = {mAP:.4f}")

    # Save
    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()},
               "checkpoints/st_det/best.pt")

if __name__ == "__main__":
    main()
```

### 8.2 Model soup script

```python
# scripts/build_soup.py
"""Build model soup by averaging Phase 2 specialist backbones."""

import torch
from collections import OrderedDict

# Load 3-4 specialist backbones
specs = []
for task in ["det", "act", "psr"]:
    ckpt = torch.load(f"checkpoints/st_{task}/best.pt")
    specs.append(ckpt["model_state_dict"]["backbone"])

# Average (uniform weights)
avg_state = OrderedDict()
for key in specs[0].keys():
    stacked = torch.stack([s[key].float() for s in specs])
    avg_state[key] = stacked.mean(dim=0).to(specs[0][key].dtype)

# Save
torch.save(avg_state, "checkpoints/soup_backbone.pt")
print("Soup backbone saved")
```

### 8.3 MTL finetune from soup (skeleton)

```python
# scripts/train_mtl_finetune.py
"""MTL finetune from soup initialization."""

import torch
from src.models.mvit_mtl_model import MTLMViTModel
from src.models.yolov8_head import YOLOv8Head
from src.models.storm_decoder import STORMDecoder
# etc.

def main():
    # Build MTL model with new architecture
    model = MTLMViTModel(
        backbone="dinov2_l_frozen",
        det_head="yolov8",
        act_head="temporal_attn",
        psr_head="storm",
        pose_head="current",
    )

    # Load soup backbone
    soup_state = torch.load("checkpoints/soup_backbone.pt")
    model.load_state_dict(soup_state, strict=False)

    # Initialize heads from Phase 2 specialists
    for task in ["det", "act", "psr"]:
        spec = torch.load(f"checkpoints/st_{task}/best.pt")
        # Load the task's head from the specialist
        # (use strict=False with shape filtering as in 187 B-9)

    # Train with Path-D fixes at low LR
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 5e-5},
        {"params": model.head.parameters(), "lr": 5e-4},
    ])
    # ... train loop with Path-D fixes ...
```

---

## 9. RISK MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Phase 2 takes longer than expected | High | Medium | Run 3 ST trainings in parallel on 2 GPUs |
| Single-task specialists underperform | Medium | High | Verify each ST works on a 1-epoch smoke first |
| Soup'd backbone is worse than individual | Low | High | Try weighted soup (e.g., 0.5 each for top 2) |
| MTL finetune overfits | Medium | Medium | Use early stopping; freeze backbone for first 5 epochs |
| Hypothesis not supported (MTL << ST) | Medium | Critical | Revert to L2+L3+method story; show pathology |

---

## 10. WHAT THIS FILE ENABLES (vs 187, 188, 189)

- **187 (status):** What's done. **188 (per-head):** What to change per head. **189 (backbone+MTL):** What to share.
- **190 (this file):** **The training sequence** that turns those changes into a paper-ready model. Without this file, the per-head and backbone changes are individual improvements; with this file, they're a coordinated plan.

**Key insight: the per-head and backbone changes are necessary, but they're not enough without proper validation. Phase 2 (ST baselines) is the gate that makes the paper's MTL claims defensible.**

---

## 11. THE COMPLETE TIMELINE (3-WEEK EXAMPLE)

```
Week 1:
  Mon-Tue:  Phase 1 — implement YOLOv8 head, STORM decoder, temporal attn activity head
  Wed:       Run 4 smoke tests (1 epoch each, ~30 min each)
  Thu-Fri:   Foundation backbone smoke test (DINOv2-L or InternVideo2-L)

Week 2:
  Mon-Wed:  Phase 2 — 4 single-task pretrainings in parallel
            (Detection ~2 days, Activity ~1 day, PSR ~2 days, Pose ~0.5 day)
  Thu:       Build model soup + initial MTL finetune (1 epoch)
  Fri:       Continue MTL finetune (5-7 epochs)

Week 3:
  Mon-Tue:  Continue MTL finetune (10-20 epochs)
  Wed:       Evaluate MTL on val + test
  Thu:       Compare MTL vs ST vs SOTA, build headline table
  Fri:       Write paper, polish, submit
```

**Tight but feasible.**

---

## 12. KEY REFERENCES

- Caruana 1997: "Multitask Learning" — original MTL paper
- Kendall et al. 2018: "Multi-Task Learning Using Uncertainty to Weigh Losses" — Kendall weighting (the pathology we just fixed)
- Chen et al. 2018: "GradNorm: Gradient Normalization for Adaptive Loss Balancing" — alternative to Kendall
- Yu et al. 2020: "Gradient Surgery for Multi-Task Learning" — PCGrad (we already use this)
- Wortsman et al. 2022: "Model Soups" — averaging weights of fine-tuned models
- Ilharco et al. 2022: "Editing Models with Task Arithmetic" — task arithmetic primitive
- Ma et al. 2018: "Modeling Task Relationships with Multi-gate Mixture-of-Experts" (MMoE)
- Schoonbeek et al. 2024: "IndustReal: A Dataset for Understanding Industrial Assembly Actions" (WACV) — primary SOTA source

---

*This file is the operational plan. Files 188 + 189 provide the *what* (per-head + backbone changes); this file provides the *when* and *how* (training sequence and validation framework). Together they form the complete path to a paper-ready MTL model that supports the hypothesis "MTL is helping, not hurting."*