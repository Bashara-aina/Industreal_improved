# 02 — Training & Loss Improvements

**Goal:** Squeeze every available point of accuracy out of the architecture from Doc 01. Most of these are config/training-loop changes — small code, large effect.

**Source files affected:** `train.py`, `losses.py`, `config.py`, `industreal_dataset.py`

---

## Diagnosis

The architectural fixes in Doc 01 set the ceiling. Training fixes determine whether you reach it. Three meta-issues drive the current training:

| Issue | Symptom | Root cause |
|---|---|---|
| **T1** No video-level pretraining | Activity Top-1 stuck near 66% | Backbone sees ImageNet only; never learns industrial visual statistics |
| **T2** All four losses fight for backbone capacity from epoch 0 | Tasks plateau before convergence | Multi-task gradient interference without staged warmup |
| **T3** Naive class balancing | Long-tail action classes (e.g. "place_motor_assembly") never get learned | CB-Focal weights are computed but rare classes still see <50 examples/epoch |

---

## A. Self-supervised pretraining of the backbone (HIGHEST LEVERAGE)

This is the single biggest unlock for Activity Top-1. The bot's analysis correctly identified that MViTv2 wins on **pretraining data**, not architecture. You can match that advantage with public weights.

### A.1 Initialize the backbone from VideoMAE V2 (Kinetics-400 pretrained)

VideoMAE V2 fine-tuned on Kinetics-400 reaches **87.4% Top-1**. The pretrained weights are public on HuggingFace (`MCG-NJU/videomae-base-finetuned-kinetics`). You can't drop a ViT into your CNN backbone slot, but you **can** use a 2-stream design:

- **Stream 1:** Your existing ResNet-50 / ConvNeXt-Tiny — handles detection, pose, PSR, head pose (per-frame tasks).
- **Stream 2:** VideoMAE V2 ViT-S/16 (small variant, 22M params) — handles **only Activity**, takes the same 16-frame clip, outputs a single 384-D feature, fused with your existing activity head before the classifier.

Only the activity head sees both streams. Detection / pose / PSR are unchanged.

```python
# In model.py, add:
class VideoMAEStream(nn.Module):
    def __init__(self, ckpt='MCG-NJU/videomae-small-finetuned-kinetics'):
        super().__init__()
        from transformers import VideoMAEModel
        self.encoder = VideoMAEModel.from_pretrained(ckpt)
        # Freeze for first 10 epochs, then unfreeze with low LR
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, clip):  # clip: [B, T=16, 3, 224, 224]
        out = self.encoder(clip)  # [B, num_patches, hidden]
        return out.last_hidden_state.mean(dim=1)  # [B, 384]
```

In `ActivityHead`, change the input dim of `proj_features` to include this 384-D feature:
```python
proj_input_dim = det_conf_size + c5_channels + p4_channels + 384  # was 2328, now 2712
```

**Cost:** +22M params, +6 GFLOPs (small variant). Inference FPS drops ~25% but stays >15 FPS on RTX 3060.

**Expected gain on Activity Top-1: +5 to +7%.** This is the biggest single intervention in this entire plan.

### A.2 If A.1 is too heavy — masked image pretraining on IndustReal

If the 2-stream design is too expensive, do a cheap alternative:
1. Take all IndustReal RGB frames (no labels).
2. Run 30 epochs of MAE-style masked image modeling on your existing backbone.
3. Use those weights as initialization for the multi-task training.

This is what the IndustReal authors call "domain adaptive pretraining" — gives roughly half the gain of A.1 (+2 to +3% Top-1) at much lower cost. Use the `MaskedImageModeling` recipe from `timm`.

---

## B. Staged training schedule (instead of all-tasks-from-scratch)

Right now `train.py` uses Kendall weighting from epoch 0 with all 4 losses active. This is the textbook recipe but it's suboptimal here because the **detection head is already initialized poorly** (random) while the backbone is ImageNet-pretrained — the gradients from the random head corrupt the good backbone features.

### B.1 Three-stage training schedule

```
Stage 1 (epochs 1-5): Detection-only warmup
  - Active losses: L_det only
  - Backbone: layer1-3 frozen, layer4 + FPN + det head trainable
  - Goal: bring det head to ~70% mAP before exposing the backbone to other tasks
  - LR: 5e-4 (high), warmup 2 epochs

Stage 2 (epochs 6-15): Add pose + head pose
  - Active losses: L_det + L_pose + L_head_pose, all Kendall-weighted
  - Backbone: layer3+4 + FPN + det head + pose head + head_pose_head trainable
  - Activity and PSR heads exist but are NOT in the loss yet
  - LR: 2e-4

Stage 3 (epochs 16-100): Full multi-task with EMA
  - Active losses: L_det + L_pose + L_head_pose + L_act (ramped) + L_psr
  - All parameters trainable
  - EMA decay 0.999 starts here (don't EMA earlier — backbone is still moving fast)
  - LR: 1e-4 with cosine annealing
```

**Action:** In `train.py`, add a `stage` variable computed from epoch:
```python
def get_stage(epoch):
    if epoch <= 5:   return 1
    if epoch <= 15:  return 2
    return 3

# In the training step:
stage = get_stage(epoch)
loss_det = compute_det_loss(...)
loss = loss_det
if stage >= 2:
    loss = loss + kendall_weight_pose * compute_pose_loss(...)
    loss = loss + kendall_weight_head * compute_head_pose_loss(...)
if stage >= 3:
    act_ramp = min(1.0, (epoch - 15) / 5.0)  # ramp Activity in stage 3
    loss = loss + kendall_weight_act * act_ramp * compute_act_loss(...)
    loss = loss + kendall_weight_psr * compute_psr_loss(...)
```

**Expected impact:** Faster convergence (val metrics peak by epoch 50 instead of 80), and a +1 to +2% Top-1 / +1 mAP improvement at convergence vs flat schedule.

### B.2 Differential learning rates

The backbone (pretrained) and the heads (random) should not share an LR. Standard fine-tuning practice:

```python
# In train.py optimizer setup:
backbone_params = list(model.backbone.parameters()) + list(model.fpn.parameters())
head_params = [p for n, p in model.named_parameters()
               if not n.startswith('backbone') and not n.startswith('fpn')]

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': C.BASE_LR * 0.1},   # 1e-5 for backbone
    {'params': head_params,     'lr': C.BASE_LR},         # 1e-4 for heads
], weight_decay=C.WEIGHT_DECAY)
```

Saves the pretrained features from being clobbered by random-head gradients. Expected: **+0.5 to +1% across all tasks**.

---

## C. Loss function fixes

### C.1 Detection: replace SmoothL1 with GIoU loss for box regression

`losses.py` `FocalLoss.forward` uses `F.smooth_l1_loss` for the regression target. Modern detectors (YOLOv8, DETR) all use **GIoU / DIoU / CIoU** which directly optimize the IoU metric you're evaluated on. SmoothL1 optimizes a proxy.

```python
# In losses.py FocalLoss.forward, replace the smooth_l1_loss block with:
if pos_mask.sum() > 0:
    pred_boxes = self._decode(anchors[pos_mask], reg_preds[i][pos_mask])
    gt_boxes_pos = matched_boxes[pos_mask]
    from torchvision.ops import generalized_box_iou_loss
    total_reg = total_reg + generalized_box_iou_loss(
        pred_boxes, gt_boxes_pos, reduction='sum'
    ) / num_pos
```

Add a `_decode` static method that mirrors `POPWMultiTaskModel._decode_boxes`. Expected gain: **+1.5 to +2 mAP@0.5**.

### C.2 Activity: replace CB-Focal with LDAM-DRW

CB-Focal weights the loss by inverse class frequency, but for IndustReal's tail classes (some have <30 samples), it still struggles. **LDAM (Label-Distribution-Aware Margin) + DRW (Deferred Re-Weighting)** is the current SOTA for long-tail classification on small datasets — used by recent fine-grained action recognition papers.

```python
# In losses.py, add:
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.weight = weight  # Use CB weights here, set after epoch 60 (DRW)

    def forward(self, logits, target):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        m_list = self.m_list.to(logits.device)
        batch_m = m_list[target].view(-1, 1)
        x_m = logits - batch_m * index.float()
        return F.cross_entropy(self.s * x_m, target, weight=self.weight)
```

In `train.py`, swap `ClassBalancedFocalLoss` for `LDAMLoss`. After epoch 60, set `loss_fn.weight = cb_weights` (this is the DRW step — defer the re-weighting until features are stable).

Expected gain: **+1.5 to +2.5% Top-1**, especially on the long-tail classes that dominate macro-F1.

### C.3 PSR: focal-style loss, not BCE

`losses.py` likely uses `BCEWithLogitsLoss` for PSR. PSR has heavy class imbalance per component (component appears in <30% of frames typically). Use **binary focal loss** instead:

```python
def binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()
```

Expected gain on PSR F1: **+0.5 to +1**.

### C.4 Pose: confidence-weighted Wing loss for hand joints

IndustReal hands.csv has many tracking errors (your `visualize_filtered.py` confirms this — joints fall on shadows). Currently `WingLoss.forward` accepts an optional `weight` but it's not being used in the pose loss path. Pass per-joint confidence to weight the loss:

```python
# In train.py compute_pose_loss:
weight = pose_confidence_target  # [B, 17] from dataset, valid joints only
loss_pose = wing_loss(pred_keypoints, target_keypoints, weight=weight)
```

This stops the model from regressing toward the broken joint coordinates. Expected: cleaner pose features, indirect +0.3% Activity Top-1 (via better PoseFiLM).

---

## D. Augmentation upgrades

### D.1 RandAugment for the backbone (not for keypoints)

Your `industreal_dataset.py` uses spatial augmentation (flip + crop). Add **photometric** RandAugment to harden the backbone against IndustReal's lighting variations (the dataset has both natural and simulated lighting).

Use `torchvision.transforms.v2.RandAugment(num_ops=2, magnitude=9)` *only* on the image — never on keypoint targets (would break geometry). Photometric ops only: brightness, contrast, posterize, solarize, sharpness, equalize. **Skip** translate/rotate/shear (those need keypoint co-transforms).

Expected: +0.5 mAP, +0.5% Top-1.

### D.2 CutMix for activity (not just Mixup)

Mixup blends pixels globally. **CutMix** pastes a rectangular patch from one video into another — better for fine-grained recognition because the model has to figure out which region drives the label. Industry-standard for action recognition since 2020.

```python
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H); y2 = np.clip(cy + cut_h // 2, 0, H)
    rand_idx = torch.randperm(B, device=x.device)
    x[:, :, y1:y2, x1:x2] = x[rand_idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[rand_idx], lam
```

In `train.py`, alternate epochs: even epochs use Mixup, odd use CutMix. Expected: +0.5–1% Top-1 over Mixup-only.

### D.3 Temporal augmentation: random frame stride

Currently `TRAIN_FRAME_STRIDE = 5` (or 3 after Doc 01.A.2). Vary it per clip: pick a random stride in `{2, 3, 4, 5}`. Forces temporal invariance. Negligible cost. Expected: +0.3% Top-1.

---

## E. Optimizer & schedule

### E.1 Replace AdamW with Lion or Lamb

For multi-task transformer-style models, **Lion** (Chen et al., 2023) consistently outperforms AdamW by 0.5–1% with **half the memory** for optimizer state. Drop-in replacement:

```python
# pip install lion-pytorch
from lion_pytorch import Lion
optimizer = Lion(model.parameters(), lr=C.BASE_LR * 0.3, weight_decay=C.WEIGHT_DECAY * 3)
# Lion needs LR ~3× smaller and WD ~3× larger than AdamW
```

Memory saved (~1 GB on RTX 3060) lets you bump batch size from 2 to 3, which is itself worth +0.3% Top-1 from better gradient stats.

### E.2 OneCycleLR with super-convergence

Cosine annealing with warmup restarts (T_0=10, T_mult=2) is fine but slow. **OneCycleLR** with high peak LR (5e-4) + aggressive cosine decay converges 30% faster:

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[5e-5, 5e-4],  # one per param group (backbone, heads)
    epochs=C.EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos',
)
```

Especially useful given your patience=10 — you want to peak early.

### E.3 Stochastic Weight Averaging at the end

After regular training, run **5–10 epochs of SWA** at a constant low LR (1e-5) and average the weights. This is +0.3 to +0.5% across all tasks for free, *after* training.

```python
# At end of train.py:
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
for epoch in range(C.EPOCHS, C.EPOCHS + 8):
    train_one_epoch(...)
    swa_model.update_parameters(model)
    swa_scheduler.step()
update_bn(train_loader, swa_model)  # recalibrate BN stats
```

---

## F. Validation-time test-time augmentation (TTA)

Free accuracy at evaluation, no retraining needed.

### F.1 Horizontal flip TTA

Run inference twice — original + horizontal flip — and average logits. For detection, flip predicted boxes back. For pose, swap left/right keypoints.

```python
# In evaluate.py:
out1 = model(image)
out2 = model(torch.flip(image, dims=[3]))
# Flip keypoint indices for pose:
out2['keypoints'] = flip_keypoints(out2['keypoints'], pairs=C.KEYPOINT_FLIP_PAIRS)
# Average activity logits:
act_logits = 0.5 * (out1['act_logits'] + out2['act_logits'])
```

Expected: +0.5% Top-1, +0.3 mAP. **Doubles inference time**, so use only at val/test, not deployment.

### F.2 Multi-crop TTA for Activity

Take 5 crops (4 corners + center) at training resolution, average activity logits. Standard MViTv2 evaluation does this. Expected: +1% Top-1.

---

## Summary table — training-side gains

| Improvement | Effort | Expected gain | Notes |
|---|---|---|---|
| **A.1 VideoMAE V2 stream** | high | **+5 to +7% Activity Top-1** | Single biggest unlock |
| A.2 MAE pretraining (alternative) | medium | +2 to +3% Top-1 | If A.1 too expensive |
| B.1 Staged training | low | +1 to +2% across tasks | Quick win |
| B.2 Differential LR | trivial | +0.5 to +1% all | Always do this |
| C.1 GIoU regression | low | +1.5 to +2 mAP | Always do this |
| C.2 LDAM-DRW | medium | +1.5 to +2.5% Top-1 | Long-tail fix |
| C.3 Focal PSR loss | trivial | +0.5 to +1 F1 | |
| C.4 Confidence-weighted Wing | trivial | indirect +0.3% Top-1 | Cleaner features |
| D.1 RandAugment | low | +0.5 mAP | |
| D.2 CutMix | low | +0.5 to +1% Top-1 | |
| D.3 Random temporal stride | trivial | +0.3% Top-1 | |
| E.1 Lion optimizer | trivial | +0.5–1% + memory | Frees batch size headroom |
| E.2 OneCycleLR | low | faster convergence | |
| E.3 SWA at end | low | +0.3 to +0.5% | Free win |
| F.1 Flip TTA | trivial (eval) | +0.5% Top-1 | Eval only |
| F.2 5-crop TTA | low (eval) | +1% Top-1 | Eval only |

---

## Implementation order

1. **B.2 + E.1 + C.1** (~half day): Quick wins. Differential LR, Lion, GIoU.
2. **C.2 + C.3** (~half day): LDAM-DRW activity loss, focal PSR.
3. **B.1** (~1 day): Staged training schedule.
4. **D.1 + D.2 + D.3** (~half day): Augmentation upgrades.
5. **A.1** (~3 days): VideoMAE V2 integration. Biggest gain — do it last because it's the largest code change.
6. **E.3 + F.1 + F.2** (~half day): SWA + TTA, applied after training is finalized.
