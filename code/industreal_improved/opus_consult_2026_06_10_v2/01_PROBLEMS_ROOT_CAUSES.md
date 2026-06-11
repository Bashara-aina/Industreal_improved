# Root Causes, Problems & Hypotheses — Complete Catalogue
## POPW Opus Consultation v2 (2026-06-11)

---

## Executive Summary

The POPW model has **never produced trustworthy multi-task metrics**. A deep forensic audit (conducted June 10, 2026) identified **24 root causes** (RC-1 through RC-24) spanning the entire training pipeline: EMA corruption, broken reinitialization, label corruption, input mismatches, dead temporal machinery, inverted attention scaling, and insufficient data. **Every post-retrain metric in our evidence set is a measurement of a corrupted blend**, not the trained model.

---

## 1. Critical Root Causes (Ship-Blockers)

### RC-13: EMA Shadow Never Reset + Collapsed Shadow Restored + best.pth = EMA
**Severity: CRITICAL — Invalidates ALL post-retrain evidence**

**Location**: `train.py:2636`, `train.py:2650-2655`, `train.py:3433-3437`

**Mechanism**:
1. `_reinit_dead_heads` at `train.py:2636` does: `ema.shadow[_n] = ema.shadow[_n].clone().detach()` — this is a **no-op** that copies the old shadow into itself
2. Three lines later at `train.py:2650-2655`, the checkpoint's epoch-43 `ema_shadow` (the **collapsed weights**) is restored over the shadow
3. `best.pth` saves EMA weights (`train.py:3433-3437`), not the trained weights
4. With 3,112 train samples at bs=2 ≈ 1,556 EMA updates at decay 0.999: saved checkpoint ≈ `0.999^1556 ≈ 21%` collapsed-old weights

**Evidence**:
- `train.log:441` — val swapped to EMA weights
- `train.log:495-497` — EMA model predicting class 28 at 66.6% (pre-patch collapse signature)
- `train.log:606-608` — raw model predicting class 8 (different from EMA)
- `train.log:649` — `EMA vs Raw delta — psr_f1=-0.0909`: raw model HAD psr_f1=0.0909; EMA destroyed it
- `eval.log:22-31` — `score_p50 ≈ 0.0 / 1e-39` on every batch (collapsed head signature)

**Fix**: P1 — Copy `param.data` into shadow (not shadow into shadow). P2 — Set `USE_EMA=False` for recovery run.

---

### RC-14: Detection Reinit Misses the Trunk
**Severity: HIGH — Detection cannot recover**

**Location**: `train.py:1623` vs `model.py:508-509`

**Mechanism**: `_reinit_dead_heads` iterates `('cls_tower', 'reg_tower')` but the model's modules are named `cls_subnet`/`reg_subnet`. Only the two final 3×3 convs (`cls_score`, `reg_pred`) were reset. The 4-layer conv trunk that feeds them was NEVER re-initialized — it still contains collapsed features from epoch 43.

**Evidence**: Fresh 0.01-std final conv on top of collapsed trunk → bimodal score distribution at eval (median 1e-39, max 0.97). The trunk emits huge-magnitude features that overwhelm the fresh final layer.

**Fix**: P3 — Change `for tower_attr in ('cls_subnet', 'reg_subnet'):` at `train.py:1623`.

---

### RC-15: Mixup/CutMix Corrupt Activity Labels
**Severity: HIGH — Activity head trains on wrong labels**

**Location**: `train.py:377-486`, `losses.py:491-495`

**Mechanism**:
1. `mixup_activity`/`cutmix_activity` blend `act_logits` AFTER the forward pass (`train.py:407, 470`)
2. The model NEVER sees mixed inputs — `images_mixed` is constructed but never fed to the model
3. The mixed soft target is argmax'd by LDAM (`losses.py:491-495`): when `lam < 0.5`, the loss supervises frame i's logits with frame j's label
4. CutMix has no `0.3 ≤ lam ≤ 0.7` gate, uses `Beta(1,1)=U(0,1)`, was active for the whole retrain
5. With bs=2, this is a coin-flip label swap on a huge share of activity batches

**Evidence**: A fresh head trained one epoch under ~50% label noise on 4 recordings WILL collapse to a constant class.

**Fix**: P4 — `USE_MIXUP=False`, `CUTMIX_ALPHA=0.0` until implementation actually mixes images before forward.

---

### RC-16: Inverted Attention Scaling in Activity ViT
**Severity: MEDIUM-HIGH — Attention saturates to one-hot**

**Location**: `model.py:1097-1098`

**Mechanism**: `scale = self.head_dim ** -0.5; attn = torch.matmul(q, kᵀ) / scale` — dividing by `d^-0.5` multiplies attention logits by `√d = 8` instead of dividing. Logits are **64× larger** than standard. After any training, softmax saturates to near-one-hot → gradients through attention vanish.

**Impact**: Affects ONLY the activity head. PSR uses `nn.TransformerEncoder` (correct internal scaling). Pose heads are MLPs.

**Fix**: P5 — `attn = torch.matmul(q, k.transpose(-2, -1)) * scale`

---

### RC-17: Train/Eval Input Mismatch on VideoMAE Half
**Severity: HIGH — Eval measures a different model than training**

**Location**: `eval_post_reinit.py:63`, `industreal_dataset.py:1299-1383`, `model.py:1347-1348`

**Mechanism**: Training passes real clips via `collate_fn` (includes `clip_rgb`). Standalone eval uses `collate_fn_sequences` which omits `clip_rgb` entirely → `model.forward` gets `clip_rgb=None` → `feat = cat([feat, zeros_like(feat)])`. **Half the classifier input is zeroed at eval but real in training.**

**Evidence**: Both "baseline" and "post-retrain" evals share this flaw. Explains part of the EMA-val (class 28, clips present) vs standalone-eval (class 8, clips zeroed) discrepancy.

**Fix**: P6 — `collate_fn = _ds_module.collate_fn` (val dataset is never in sequence mode here).

---

### RC-18: FeatureBank is Dead (Always Returns Current Frame ×16)
**Severity: HIGH — Temporal machinery contributes nothing**

**Location**: `model.py:1148-1150`, `train.py:975,1128`, `evaluate.py:2876`

**Mechanism**: Every call site invokes `model(images, clip_rgb=...)` with `video_ids=None`, so `FeatureBank.forward` takes the fallback and returns the **current frame replicated 16×**. The TCN+2×ViT therefore process 17 near-identical tokens; the activity head is effectively a per-frame MLP. The "T=16 temporal context" does not exist at runtime.

**Fix**: P11 (defer) — Pass `video_ids` to engage the FeatureBank. Only after heads are alive.

---

### RC-19: det_conf is Raw Unbounded Logits — Couples Det Collapse into Activity
**Severity: HIGH — Activity collapse is downstream of detection**

**Location**: `model.py:1944-1945, 1968-1972`

**Mechanism**: `cls_preds.max(dim=1)[0]` takes max raw logits (not probabilities) and concatenates with GAP features (~O(0.1–1)). With collapsed det trunk, the 24 `det_conf` dims have enormous, frame-invariant magnitude. Activity input is constant at L2 243.39 ± 0.001.

**Fix**: P7 — `det_conf = torch.sigmoid(cls_preds).max(dim=1)[0]`

---

## 2. Secondary Root Causes

### RC-20: Combined Metric is Pose-Only in Practice
**Location**: `train.py:123-126, 1774-1789`

`combined = 0.30·mAP50 + 0.35·act_f1 + 0.15·(1/(1+head_pose_MAE)) + 0.20·psr_f1`

With det=0, act=0, psr=0: `0.15/(1+0.344) = 0.1116`. Checkpoint selection cannot reward det/act/psr recovery.

### RC-21: MATCH_PROBE Can Never Fire
**Location**: `losses.py:230`

`probe_anchor_matching` called with `_state=None` each time → `_state` recreated per call → `n` always 1 → `1 % 200 != 0` → diagnostic never logs.

### RC-22: Anchor/GT Scale Mismatch
**Location**: `config.py:249` vs `config.py:243-247`

`ANCHOR_SIZES=(24,48,96,192,384)` but k-means on GT: w=146–594px, centers 164–404px. Only P6/P7 (1.6% of anchors) can reach IoU≥0.5 with typical GT. Also `neg_iou_thresh` never passed to FocalLoss constructor.

### RC-23: Eval Slice is Unrepresentative
**Location**: `eval_post_reinit.py`

First 200 frames sequentially (shuffle=False, 50 batches × bs 4). Contains 42 GT boxes, all one class (background), mostly-constant PSR state, narrow activity label set.

### RC-24: Training Subset Cannot Support the Task
**Location**: `--subset-ratio 0.05`

4 recordings, 3,112 frames, ~12/75 activity classes present, very few GT-box frames. Even a bug-free run cannot learn a 75-way classifier from this.

---

## 3. Confirmed-Working Hypotheses (Rejected)

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| H1.1: Anchor format mismatch | **Rejected** | Both model and targets are anchor-based, pixel xyxy |
| H1.2: Box decoder bug | **Rejected** | decode(encode(gt)) == gt verified |
| H1.3: NMS too aggressive | **Rejected** | DET_PROBE bypasses NMS, still mAP=0 |
| H1.4: GT in wrong format | **Rejected** | COCO xywh→xyxy conversion verified |
| H1.5: Anchor/target coordinate space | **Rejected** | Both pixel-space, IoU-invariant normalization |
| H2.1: Activity learned the prior | **Rejected** | Class 8 is NOT the prior's argmax |
| H2.3: Attention collapse (generic) | **Partially confirmed** | RC-16 provides the specific mechanism |
| H2.4: WeightedRandomSampler bias | **Real but secondary** | 12-class pool on 5% subset |
| H3.1: Pose was re-initialized | **Refuted** | `_reinit_dead_heads` contains NO pose tensors |

---

## 4. The PSR "Recovery" That Wasn't

The "8× PSR recovery" reported in earlier analysis is an **artifact**:
- `eval.log:73` — PSR still produces **1 unique binary pattern** across all 200 frames
- The constant pattern is `[1,0,0,...,0]`
- The eval slice (first 200 frames, sequential) has comp0=1 nearly everywhere (train prevalence = 1.0)
- Constant prediction ∩ skewed slice = edit_score 0.73 and comp0 F1 = 1.0
- This is NOT recovery — it's a measurement artifact

---

## 5. The Activity "Top-5 = 0.06" Explained

- `act_top5 = 0.06` is NOT "below random" or "systematic exclusion"
- Constant logits give a fixed top-5 set {8, 73, 74, 71, 70}
- 12 of 200 GT labels fall in that set
- The right reference is the GT marginal mass of that fixed set
- The comparison to 1/15 random is meaningless

---

## 6. Structural Problems Beyond Bugs

### 6.1 Architecture Cannot Win on Activity
- Per-frame ConvNeXt + 2 tiny ViT blocks over a fake bank will NOT reach 65% Top-1
- The benchmark is **clip-level Top-1 over 16 uniform frames per action segment**
- We currently train per-frame on NA-dominated labels and evaluate frame-level — protocol mismatch
- Need: fine-tuned K400 video encoder + clip-level training + protocol alignment

### 6.2 Detection Design Mismatch
- ASD is NOT COCO. It's 0–3 large objects (146–594px) where the hard part is fine-grained STATE
- Dense 24-class anchors at 173k locations is the wrong shape
- Need: class-agnostic localizer + ROI-Align state classifier

### 6.3 PSR Loss Design
- Per-frame BCE on 95%-static labels teaches the constant pattern
- Need: predict TRANSITIONS with monotonic state accumulator, not per-frame binaries
- B2 baseline (F1=0.731) is ASD-confidence accumulation + procedure-order constraints — barely neural

### 6.4 Data Starvation
- 5% subset = 4 recordings = 12/75 classes = structural ceiling
- Need: at minimum 25% subset for meaningful training
- Synthetic data pretraining wired but unused

### 6.5 Defensive Machinery Hides Bugs
- ~15 layers of NaN guards, smooth caps, sensitivity penalties, staged Kendall zeroing
- These don't fix bugs — they HIDE them and distort gradients
- Need: assert-and-crash, fixed per-task weights, plain CE+label_smoothing

---

## 7. Open Questions for Opus

1. **Given all 24 root causes, what is the optimal order to fix them?** Some are zero-cost (eval fixes), some require retraining.

2. **Should we keep the current architecture or redesign from scratch?** We are open to changing backbone (ViT, Swin, EfficientNet), changing heads, changing the entire training flow.

3. **What is the most effective multi-task training strategy?** Staged? Joint? Two-stage with frozen backbone cache? Curriculum learning?

4. **How do we prevent catastrophic forgetting between heads?** When one head collapses, it drags others down through shared backbone gradients.

5. **What is the minimum viable experiment to prove the architecture can learn?** We need a smoke test that convincingly shows all heads can learn simultaneously.

6. **Should we use knowledge distillation from dedicated single-task models?** YOLOv8m for detection, MViTv2 for activity — distill into unified model.

7. **How do we handle the extreme class imbalance?** 75 activity classes with heavy long-tail, 11 PSR components with 95% static labels, 24 detection classes with sparse GT.

8. **What backbone would you recommend for this specific multi-task problem?** ConvNeXt-Tiny is our current choice but we're open to alternatives that better serve multi-task learning.
