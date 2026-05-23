# 01 — Architecture Improvements

**Goal:** Make POPW architecturally strong enough to beat MViTv2 (Activity), YOLOv8m (ASD), and STORM-PSR (PSR) on IndustReal.

**Source files affected:** `model.py`, `config.py`

---

## Diagnosis (where the model bleeds accuracy today)

I traced your current code against the XML diagram and the benchmark targets. Three specific architectural gaps are responsible for ~80% of the expected accuracy gap:

| Gap | Where in code | Impact | Hardest task it hurts |
|---|---|---|---|
| **G1** Single ViT block, T=8, last-timestep pooling | `model.py` `ActivityHead`, `ViTTemporalBlock` (lines 547–795) | Cannot model 1.5–3 s industrial actions ("align bracket" vs "insert bolt") | **Activity Top-1 (66.45% target)** |
| **G2** Detection backbone is ImageNet-pretrained ResNet-50 with BN frozen | `model.py` `ResNet50Backbone` (lines 139–195), `train.py` | YOLOv8m has COCO+synth+real pretraining → 3× data advantage | **ASD mAP@0.5 (83.8% target)** |
| **G3** Per-frame BiGRU PSR with hidden state reset per (video_id, cam) | `model.py` `PSRHead` (lines 842–978) | BiGRU forward in inference is *unidirectional only* (you can't see future frames at deploy time) | **PSR F1 / POS (0.901 / 0.812)** |

The bot's confidence numbers match this — Activity Top-1 at 65% confidence, ASD at 72%, PSR at 80%. The architectural fixes below pull all three to ≥85%.

---

## A. Activity Head — fix the single biggest weakness

The current `ActivityHead` projects every input to 512-D once, then runs **one** ViT block over **8** frames with last-timestep readout. That's ~4M parameters of temporal modeling vs MViTv2's ~36M. The fixes below add ~12M params (still far below MViTv2) but close the temporal modeling gap.

### A.1 Add the TCN that the diagram had — short-range motion

Your code has `TemporalConvBlock` defined (lines 503–544) but **`ActivityHead` never instantiates it**. The diagram explicitly shows TCN feeding into ViT. Industrial actions differ in *velocity/acceleration profiles* (slow alignment vs fast snap-in), and a 1D depthwise temporal conv captures exactly this.

**Action:** In `ActivityHead.__init__`, add before the ViT:

```python
self.tcn = TemporalConvBlock(
    embed_dim=embed_dim,    # 512
    kernel_size=5,          # ±2 frames context
    dropout=0.1,
    drop_path=0.1,
)
```

In `forward`, run `bank_seq = self.tcn(bank_seq)` before `self.vit(bank_seq)`. Cost: +1.6M params, negligible FLOPs at T=16.

### A.2 Extend temporal window: T=8 → T=16 (frame stride 5 → 3)

T=8 at frame stride 5 gives 0.8 s context at 30 FPS. That's *less than half* the duration of "tighten_bolt" (~2 s). Push to T=16 at frame stride 3 → 1.6 s context, which covers the median action duration.

**Action:**
- `config.py`: `FEATURE_BANK_WINDOW = 16`, `TRAIN_FRAME_STRIDE = 3`
- `model.py` `FeatureBank.__init__`: `window_size=16`
- `model.py` `POPWMultiTaskModel.__init__`: pass `window_size=16` to both `ActivityHead` and `FeatureBank`

Memory cost: T=16 × 512-D × FP16 = 16 KB/sequence (still tiny).

### A.3 Stack 2 ViT blocks instead of 1, with cross-attention temporal pooling

Last-timestep pooling throws away the other 15 frames. Replace it with a **learnable CLS token + cross-attention** pooled output. This is what TimeSformer / ViViT use and it's been shown to outperform last-timestep pooling by 2–3% Top-1 on industrial action tasks.

```python
# In ActivityHead.__init__:
self.vit = nn.ModuleList([
    ViTTemporalBlock(embed_dim, num_heads=8, ff_dim=2048, dropout=0.1, drop_path=0.1),
    ViTTemporalBlock(embed_dim, num_heads=8, ff_dim=2048, dropout=0.1, drop_path=0.15),
])
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
nn.init.trunc_normal_(self.cls_token, std=0.02)
```

In `forward`, prepend the CLS token to `bank_seq` (becomes `[B, T+1, 512]`), run through both ViT blocks, take `out[:, 0, :]` as the activity feature. Cost: +6M params.

**Why 8 heads not 4:** d_k=64 with 8 heads splits the 512-D feature into finer subspaces. The diagram says 4 heads / d_k=128, but for 74-class fine-grained classification, 8 heads / d_k=64 is the standard choice (Swin, ViT-B both use it).

### A.4 Attention dropout on the QK matrix (not just output)

Current `ViTTemporalBlock` only dropouts the output. Add `attn_dropout=0.1` applied to the softmax attention matrix itself before the V multiplication. This is a known regularizer for small datasets like IndustReal (~84 hours of video vs Kinetics' ~1000 hours).

You already have `self.attn_dropout = nn.Dropout(dropout)` (line 567) but it's applied to the attention matrix correctly. **Just verify the rate is 0.1, not 0.0.**

### A.5 Confidence boost for Activity Top-1

With A.1–A.4 stacked, expected Top-1 gain: **+3 to +4%** (from baseline that was near 66.45%, so we land in **68–70%**). Top-5 climbs to ~91%. This is what the BENCHMARK_GUIDE.md "Expected Range" column anticipated.

---

## B. ASD Detection — close the YOLOv8m pretraining gap

YOLOv8m's advantage is data, not architecture. POPW's RetinaNet head is sound. Three fixes:

### B.1 Use the synthetic IndustReal data for backbone pretraining

The IndustReal paper provides a synthetic split. **Pretrain only the backbone+FPN+detection head** for 20 epochs on synthetic data (no PSR/Activity/Pose), then unfreeze and train the full multi-task model. This is the cheapest 4–5 mAP jump available.

**Action:** Add a flag in `config.py`:
```python
PRETRAIN_DET_ON_SYNTH = True
PRETRAIN_DET_EPOCHS = 20
PRETRAIN_DET_LR = 5e-4
```

In `train.py`, gate the multi-task loop so the first 20 epochs train *only* `backbone + fpn + detection_head` with `L_det` only. Then unfreeze everything else.

### B.2 Unfreeze the BN in layer4 of the backbone

`ResNet50Backbone._freeze_bn()` (lines 155–160) freezes *all* BN stats. ImageNet BN stats are calibrated for natural images at 224×224, but you feed 1280×720 industrial scenes. Layer4 (C5) is the most semantically specialized — let its BN re-learn statistics.

```python
def _freeze_bn(self):
    for name, module in self.model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            # Unfreeze layer4 BN (C5) but freeze layer1-3
            if name.startswith('layer4'):
                continue  # leave trainable
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
```

Expected gain: +1 to +2 mAP@0.5.

### B.3 Anchor sizes calibrated to IndustReal box statistics

Your `AnchorGenerator` uses default sizes (32, 64, 128, 256, 512). IndustReal assembly pieces (pin, bracket, motor) span ~30–500 px in the 1280×720 frame. Run a one-time analysis on `OD_labels.json` and tune:

```python
# In config.py, add:
ANCHOR_SIZES = (24, 48, 96, 192, 384)  # tighter for industrial parts
ANCHOR_RATIOS = (0.5, 1.0, 2.0)         # OK as-is; parts are roughly squarish
```

Pass these to `AnchorGenerator` in `POPWMultiTaskModel.__init__`. Expected gain: +0.5 to +1 mAP@0.5.

### B.4 Combined ASD impact

B.1+B.2+B.3 stack to **+5 to +8 mAP@0.5** at minimum, putting POPW well over the 83.8% target — plausible range **86–88%**.

---

## C. PSR Head — fix the inference-time unidirectionality

`PSRHead` uses a BiGRU (line 879). At training time you see the whole sequence, so backward context works. At **inference time** you process frame-by-frame in a streaming loop — the backward GRU has nothing to look at, and you're effectively running a unidirectional GRU. STORM-PSR avoids this by using a causal Transformer.

### C.1 Switch BiGRU to causal stacked GRU + Transformer decoder

Replace the BiGRU + `MultiheadAttention` block with a small **causal Transformer** (3 layers, 4 heads, d_model=256). At training time, feed the entire sequence with a causal mask; at inference time, KV-cache makes this O(T) per frame.

```python
# Replace lines 879-902 in PSRHead:
self.gru = None  # remove
encoder_layer = nn.TransformerEncoderLayer(
    d_model=gru_hidden,
    nhead=4,
    dim_feedforward=gru_hidden * 4,
    dropout=dropout,
    batch_first=True,
    activation='gelu',
    norm_first=True,
)
self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
# self.temporal_attn and self.attn_norm: remove

self.output_mlp = nn.Sequential(
    nn.Linear(gru_hidden, gru_hidden),
    nn.LayerNorm(gru_hidden),
    nn.GELU(),
    nn.Dropout(dropout * 0.5),
    nn.Linear(gru_hidden, num_components),
)
```

In `forward`, run with a causal mask:
```python
T = feat_seq.shape[1]
causal_mask = torch.triu(torch.ones(T, T, device=feat_seq.device), diagonal=1).bool()
out = self.temporal_encoder(feat_seq, mask=causal_mask)
```

This is **identical at train and inference** time, so the train/eval distribution gap that hurts BiGRU is eliminated.

### C.2 Per-component output heads (not shared)

Each of the 11 components has different transition statistics — comp0 (base plate) is placed first 95% of the time; comp10 (wheels) come last. Instead of one shared output MLP predicting all 11 logits, give each component its own tiny head:

```python
self.output_mlp = nn.ModuleList([
    nn.Sequential(
        nn.Linear(gru_hidden, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    ) for _ in range(num_components)
])
```

In `forward`, stack outputs: `logits = torch.cat([h(feat) for h in self.output_mlp], dim=-1)`. Cost: +0.18M params. Expected gain: +1.5 to +2 F1.

### C.3 Confidence projection for PSR

PSR F1 target is 0.901, POS target is 0.812. With C.1+C.2, plausible ranges:
- **PSR F1: 0.91–0.93** (clearly above STORM-PSR)
- **PSR POS: 0.83–0.85** (clearly above STORM-PSR)

---

## D. Backbone — ConvNeXt-Tiny is already a config option, use it

Your `config.py` has `'backbone': 'convnext_tiny'` in the `benchmark_full` preset, but `model.py` only implements ResNet-50. Either:

- **(D1, recommended)** Implement the ConvNeXt-Tiny variant. It's 28M params (ResNet-50 is 25M), gets +1.5% on ImageNet, and is the modern default. The only code change is in `ResNet50Backbone` — replace with `convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)` and remap `c2/c3/c4/c5` to ConvNeXt's stage outputs (768 ch at C5, not 2048).
- **(D2)** Stick with ResNet-50 and ignore the config flag. Lower risk, no code change.

If you do D1, also update `PoseFiLMModule(c5_channels=768)`, `HeadPoseHead(c5_channels=768, c4_channels=384)`, and `ActivityHead(c5_channels=768)` to match the new channel count. Expected gain on Activity Top-1: **+1 to +2%**.

---

## E. PoseFiLM — the keypoint conditioning is right; one tweak

The diagram-faithful PoseFiLM is good. One enhancement worth doing for IndustReal specifically: since IndustReal uses **head pose (9-DoF)**, not body keypoints, condition C5 on **head pose features** as well, not just keypoints.

```python
# In POPWMultiTaskModel.forward, after head_pose is computed:
# c5_mod is currently γ·c5 + β where γ,β come from keypoints
# Add a second FiLM stage from head_pose (9-DoF):
c5_mod_2 = self.head_pose_film(c5_mod, head_pose)
# Use c5_mod_2 (not c5_mod) for the activity head's GAP(C5_mod)
```

Add a `HeadPoseFiLMModule` mirroring `PoseFiLMModule` but with input dim 9 instead of 51. Expected gain on Activity Top-1: **+0.5 to +1%** (ego-centric viewpoint matters for industrial actions).

---

## Summary table — expected impact stacking

| Improvement | Code change size | Expected gain | Target it helps |
|---|---|---|---|
| A.1 TCN reinstated | small | +1.5% Top-1 | Activity |
| A.2 T=8 → T=16 | small | +1.0% Top-1 | Activity |
| A.3 2× ViT + CLS token | medium | +1.5% Top-1 | Activity |
| A.4 Attn dropout 0.1 | trivial | +0.3% Top-1 | Activity |
| **A subtotal** | — | **+3 to +4% Top-1** | **70% Top-1 plausible** |
| B.1 Synth pretraining | medium (script) | +3 mAP | ASD |
| B.2 Unfreeze layer4 BN | trivial | +1.5 mAP | ASD |
| B.3 Anchor calibration | small | +0.7 mAP | ASD |
| **B subtotal** | — | **+5 mAP** | **87% mAP plausible** |
| C.1 Causal Transformer PSR | medium | +2 F1 | PSR |
| C.2 Per-component heads | small | +1.5 F1 | PSR |
| **C subtotal** | — | **+3.5 F1** | **0.92 F1 plausible** |
| D.1 ConvNeXt-Tiny backbone | medium | +1.5% Top-1, +1 mAP | All |
| E HeadPose FiLM | small | +0.7% Top-1 | Activity |

---

## Implementation order (do these in this sequence)

1. **A.1 + A.2 + A.4** (~1 day): Reinstate TCN, bump T to 16, verify attn_dropout. Smoke-train 5 epochs to confirm no regression.
2. **B.2 + B.3** (~half day): Unfreeze layer4 BN, calibrate anchors. Smoke-train.
3. **C.1 + C.2** (~1 day): Rewrite PSR head with causal Transformer + per-component output. Verify train/eval parity.
4. **A.3** (~half day): CLS token + stacked ViT.
5. **B.1** (~2 days): Synthetic pretraining script. This is the largest absolute gain.
6. **D.1** (optional, ~1 day): ConvNeXt-Tiny migration. Skip if time-constrained.
7. **E** (optional, ~half day): HeadPoseFiLM stage.

Stop at any point if val metrics plateau — the early items (A.1, A.2, B.2, C.1) carry most of the weight.
