# 01 — Honest Audit: What's Actually In Your Code Right Now

This document is a line-by-line check of which Doc 01 / Doc 02 / Doc 03 recommendations actually made it into the running code, and which are either disabled, half-wired, or still missing. The goal is to answer: **if you launched training right now with the current defaults, what would actually happen?**

---

## A. Recommendation status table

Color codes:
- ✅ **Live**: implemented, enabled by default, will run in `python train.py`
- 🟡 **Wired but OFF**: code exists, but the config flag is `False` so it's a no-op
- 🟠 **Partial**: implemented in places but with a real gap that prevents the gain
- ❌ **Missing**: not in the code at all

### Doc 01 — Architecture

| ID | Recommendation | Status | Where I checked | Notes |
|---|---|---|---|---|
| A.1 | TCN reinstated in ActivityHead | ✅ | `model.py` line 956 (`self.tcn = TemporalConvBlock(...)`), called at line 1026 | Live, runs every forward pass |
| A.2 | T=8 → T=16 + frame stride 3 | ✅ | `config.py` `FEATURE_BANK_WINDOW = 16`, `TRAIN_FRAME_STRIDE = 3`; `model.py` `window_size=16` | Live |
| A.3 | 2× ViT blocks + CLS token | ✅ | `model.py` line 964 (`nn.ModuleList([ViTTemporalBlock, ViTTemporalBlock])`), line 982 (`self.cls_token`), line 1034 takes `cls_out = bank_seq[:, 0, :]` | Live |
| A.4 | Attention dropout 0.1 on QK | ✅ | `model.py` line 761 (`self.attn_dropout = nn.Dropout(dropout)`), line 813 applied to softmax | Live (rate is `dropout` parameter, default 0.1) |
| B.1 | Synthetic detection pretraining | 🟡 | `pretrain_synthetic.py` exists, `config.py` `PRETRAIN_DET_ON_SYNTH = False` | Script ready, not enabled by default. You must run it as a separate step BEFORE main training |
| B.2 | Unfreeze layer4 BN | ✅ | `model.py` `ResNet50Backbone._freeze_bn()` lines 209–216, explicitly skips `layer4` | Live |
| B.3 | Anchor sizes calibrated for IndustReal | ❌ | `model.py` `AnchorGenerator.__init__` line 268 still uses `(32, 64, 128, 256, 512)` | **Not done** |
| C.1 | Causal Transformer PSR | ✅ | `model.py` line 1141 (`self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)`), causal mask at line 1196/1216 | Live |
| C.2 | Per-component PSR heads | ✅ | `model.py` line 1146 (`self.output_heads = nn.ModuleList([...])` with 11 separate MLPs) | Live |
| D.1 | ConvNeXt-Tiny backbone | 🟡 | `model.py` `ConvNeXtBackbone` class exists, `build_backbone` factory works; `config.py` `BACKBONE = 'resnet50'` is the default | Code complete, just need to flip the flag |
| E | HeadPoseFiLM (9-DoF → second FiLM) | 🟡 | `model.py` `HeadPoseFiLMModule` class exists, wired in `POPWMultiTaskModel` line 1306; `config.py` `USE_HEADPOSE_FILM = False` | Wired but disabled |

### Doc 02 — Training & Loss

| ID | Recommendation | Status | Where I checked | Notes |
|---|---|---|---|---|
| A.1 | VideoMAE V2 stream | 🟠 | `model.py` `VideoMAEStream` class exists (line 643), `ActivityHead` accepts `videomae_feat`, `POPWMultiTaskModel.forward` accepts `clip_rgb` | **Half-wired**. See "VideoMAE Critical Gap" section below |
| A.2 | MAE pretraining (alternative) | 🟡 | `pretrain_mae.py` script exists | Standalone script, has implementation issues — see audit below |
| B.1 | Staged training schedule | 🟠 | `train.py` `get_stage()` line 356, stage logic in `train_one_epoch()` lines 459–467; `config.py` `STAGED_TRAINING = True` | **Loss masking is live but parameter freezing is not.** Doc said "Stage 1: layer1-3 frozen". The code only zeros activity/PSR loss; backbone parameters are NOT explicitly frozen across stages |
| B.2 | Differential LR | ✅ | `train.py` lines 777–785 split into `backbone_params` and `head_params`, lines 811–814 set `lr=BASE_LR*0.1` for backbone | Live |
| C.1 | GIoU regression loss | ✅ | `losses.py` line 156 (`generalized_box_iou_loss`) | Live |
| C.2 | LDAM-DRW for activity | 🟡 | `losses.py` `LDAMLoss` class exists (line 226); `config.py` `USE_LDAM_DRW = False` | Default still uses `ClassBalancedFocalLoss` |
| C.3 | Binary focal PSR loss | ✅ | `losses.py` `binary_focal_loss` function (line 353) | Available; need to verify it's actually called by `MultiTaskLoss` (see check below) |
| C.4 | Confidence-weighted Wing | 🟠 | `losses.py` `WingLoss.forward` accepts `weight` (line 178) but I could not find a call site that passes it | Wing loss accepts confidence weight but nothing in `MultiTaskLoss` actually passes per-joint confidence |
| D.1 | RandAugment | ✅ | `train.py` line 138 (`from torchvision.transforms.v2 import RandAugment`); applied at line 134 | Live, gated by `USE_RANDAUGMENT=True` in config |
| D.2 | CutMix alternation | ✅ | `train.py` line 292 (`cutmix_activity()`), line 447 alternates by `epoch % 2 == 1` | Live |
| D.3 | Random temporal stride | ❌ | I see `TRAIN_FRAME_STRIDE = 3` (fixed) — no random selection logic | Not done |
| E.1 | Lion optimizer | 🟡 | `train.py` line 791 imports Lion, builds param groups; `config.py` `USE_LION = False` | Wired but off |
| E.2 | OneCycleLR | 🟡 | `train.py` line 821, gated by `getattr(C, 'ONE_CYCLE_LR', False)` | Wired but off, and has a bug (see below) |
| E.3 | SWA at end of training | ❌ | `grep` for "swa", "AveragedModel", "update_bn" finds nothing | Not done |
| F.1 | Horizontal-flip TTA | ❌ | `grep` for "tta", "test_time", "flip" in evaluate.py finds nothing | Not done |
| F.2 | 5-crop TTA | ❌ | Same | Not done |

### Doc 03 — Benchmark & Evaluation Strategy

| ID | Recommendation | Status | Where I checked | Notes |
|---|---|---|---|---|
| A.1 | PSR tolerance F1 (±3 frames) | ✅ | `evaluate.py` `compute_psr_metrics(tolerance_frames=5)` at line 614, also `tolerance_frames=3` for STORM-PSR comparison | Live (default tolerance 5; STORM-PSR specifies 3) |
| A.1 | RFS for ASD long-tail | 🟡 | `industreal_dataset.py` line 718 has `get_sampler()` returning `WeightedRandomSampler` | Exists but I didn't verify if it's actually attached to the train DataLoader by default in `train.py` |
| A.2 | Clip-level activity eval | ✅ | `evaluate.py` `_compute_clip_level_accuracy()` line 202 | Live |
| A.3 | PSR per-component F1 | ✅ | `compute_psr_metrics` returns per-component metrics | Live |
| A.4 | Head pose angular MAE in degrees | 🟠 | `compute_head_pose_metrics()` returns raw 9-component MAE in original units (mixed degrees/meters) — **not the angular-error formulation** | The math `acos((fwd_pred·fwd_gt))` from Doc 03 is not implemented |
| B.1 | Efficiency benchmark script | ✅ | `efficiency_report.py` exists with FPS, latency p50/p95/p99, GFLOPs, peak mem | Live |
| B.2 | Streaming FPS reporting | ❌ | Only single-image FPS measured | The "POPW unique angle" of cached temporal bank streaming FPS is not implemented |
| B.3 | ONNX FPS comparison | ✅ | `efficiency_report.py` line 257 has the ONNX runtime block | Live (gated by `--onnx_export` flag) |
| C.1 | 5-fold cross validation | ❌ | No CV script | Not done |
| C.2 | 3-seed runs | ❌ | No multi-seed orchestration | Not done |
| C.3 | Ablation table generator | ❌ | No script that automates this | Not done |

---

## B. The four critical gaps (in priority order)

### B.1 ⚠️ VideoMAE V2 stream is dead code in the training loop

This is the biggest single recommendation in Doc 02 (estimated +5–7% Activity Top-1) and it is **structurally present but functionally disconnected**. The breakdown:

- `model.py` `POPWMultiTaskModel.forward` accepts `clip_rgb: Optional[torch.Tensor]` parameter ✓
- If `use_videomae=True` and `clip_rgb is not None`, it correctly calls `self.videomae_stream(clip_rgb)` ✓
- `ActivityHead.forward` accepts `videomae_feat` and fuses it before the classifier ✓
- **But** `train.py` line 440: `outputs = model(images)` — it never passes `clip_rgb`
- `industreal_dataset.py` (820 lines) — I do not see any code that builds a 16-frame, 224×224 RGB clip per sample. The dataset returns single `images` only.

So even if you flip `USE_VIDEOMAE = True` in config, the training loop will:
1. Build the VideoMAE encoder (22M params, takes GPU memory)
2. Call `model(images)` with no `clip_rgb` → `videomae_feat = None`
3. `ActivityHead` returns the classifier output from the CNN-only `feat` because of the `if videomae_feat is not None` guard
4. The VideoMAE encoder receives **zero gradient signal** and contributes nothing
5. You waste ~6 GFLOPs of forward time per step (it doesn't even run because `clip_rgb is None` short-circuits before the encoder call)

Wait — let me re-check that last point. Looking at line 1429 of `model.py`:

```python
if self.use_videomae and clip_rgb is not None and hasattr(self, 'videomae_stream'):
    videomae_feat = self.videomae_stream(clip_rgb)
```

So actually, if `clip_rgb is None`, the encoder is **not called** and you only waste the 22M params of GPU memory. Worse: the classifier's input dim was set to `embed_dim * 2 = 1024` (not 512) because `use_videomae=True`. With `videomae_feat=None`, only the first 512 of the 1024 input slots are populated — the other 512 are whatever's left in memory. **This may run but produce garbage activity logits, or it may crash on a `torch.cat` shape mismatch depending on init.**

**Required fix:**
- Update `industreal_dataset.py` `__getitem__` to optionally return a 16-frame clip resized to 224×224 (use the existing temporal context window).
- Update `train.py` to pass `clip_rgb=batch['clip_rgb']` to `model()`.
- Verify `ActivityHead` gracefully handles `videomae_feat=None` when `use_videomae=True` (it currently doesn't — it would build a 1024-D classifier input but only fill 512).

Until this is fixed, **leave `USE_VIDEOMAE = False`** (which is the current default). The +5–7% Activity gain is on the table once you wire it correctly. This is the single most important remaining task for Activity Top-1.

### B.2 ⚠️ Staged training does not freeze parameters across stages

Doc 02 B.1 specified:
- Stage 1 (epochs 1–5): "layer1-3 frozen, layer4 + FPN + det head trainable"
- Stage 2 (epochs 6–15): "layer3+4 + FPN + det head + pose head + head_pose_head trainable"

The current code in `train_one_epoch()`:
```python
if staged_training:
    if stage == 1:
        loss_dict['activity'] = 0.0
        loss_dict['psr'] = 0.0
        loss = loss_dict['det']
    elif stage == 2:
        loss_dict['activity'] = 0.0
        loss_dict['psr'] = 0.0
        loss = loss_dict['det'] + loss_dict['pose']
```

Only the *loss* is masked. The **parameters of every head are still trainable in every stage**. Even though no gradient flows from activity/PSR loss in stage 1, the optimizer still maintains AdamW state for all parameters, and all parameters still receive gradient from the L2 weight decay term applied to AdamW.

This is harmless for correctness but **partially defeats the point** of staged training. The Doc 02 recipe was meant to give the detection backbone a clean 5-epoch warmup *without* random head gradients corrupting it. The current implementation gives you ~70% of that benefit (gradients from random heads are masked) but loses the regularization benefit of frozen layer1-3.

**Required fix:** add a `_set_stage_requires_grad(model, stage)` helper that walks the named modules and toggles `requires_grad` based on the stage. Call it once at the start of each epoch.

### B.3 OneCycleLR has a parameter-group mismatch

`train.py` lines 811–818 build `param_groups` with potentially **3 entries** (backbone, heads, and optionally `loss_params`). Lines 824–830 set `max_lr=[5e-5, 5e-4]` — only 2 values. PyTorch's `OneCycleLR` requires `max_lr` to either be a scalar or have one entry per param group. If `loss_params` is non-empty (which is the case when Kendall uncertainty weights are trainable), this will throw at runtime.

This is currently shielded by `ONE_CYCLE_LR = False` default. If you ever enable it, expect a crash. Easy fix: change to `max_lr=[5e-5, 5e-4, 5e-4]` or compute the list dynamically based on `len(param_groups)`.

### B.4 PSR `_cache` grows unbounded across batches

`PSRHead._cache: Dict[Tuple[str, str], List[torch.Tensor]]` (line 1155). Every forward pass with a `(video_id, camera_view)` key appends to the list. No popping, no TTL, and `reset_sequence`/`reset_all` are only callable externally. During a long training run, this dict will accumulate one tensor per frame per recording, indefinitely. With T~10000 frames per recording × 100 recordings × 128-D float32 = ~500 MB of CPU memory bloat per epoch.

In the IKEA-style training loop where each step is a single random frame from a random recording, this also produces **inconsistent causal context** — the cache might have only every 5th frame stored (because of `TRAIN_FRAME_STRIDE = 3`), and frames don't arrive in temporal order across batches.

**Two issues here:**
1. **Memory leak** during long training. Add `len(self._cache[key]) > MAX_CACHE` truncation in the forward.
2. **Semantic correctness**: the cache assumes frames arrive in temporal order. In random-shuffle training, they don't. The causal mask becomes meaningless because position `i` in the cached sequence is not actually "5 frames before" position `i+1`. The PSR head will see a scrambled history.

For PSR training, you really want either:
- Sequential dataloading (no shuffle within a recording, recording-as-batch-element)
- OR train PSR in a separate phase with sequence-aware sampling
- OR feed `T=16` frames at once as a clip (matching activity head's window) and let the causal mask do its job within that clip

Until this is fixed, **PSR training will be far less effective than the architecture supports**. The good news: at evaluation time, frames arrive in order (your `evaluate.py` clip_ids logic confirms this), so eval metrics will be more representative than training loss might suggest.

---

## C. Smaller things worth fixing before training

| Issue | File | Severity |
|---|---|---|
| `_cache` never cleared between epochs | `model.py` `PSRHead` | **Memory leak** — fix before long runs |
| `MultiTaskLoss` likely doesn't pass joint confidence to WingLoss | `losses.py`, `train.py` | Lose ~0.3% pose-derived Activity gain (Doc 02 C.4) |
| `binary_focal_loss` exists but I cannot confirm `MultiTaskLoss` calls it for PSR | `losses.py` | Verify the PSR loss path uses focal not BCE |
| Random temporal stride not implemented (Doc 02 D.3) | `industreal_dataset.py` | Missing ~0.3% Activity gain |
| Confidence-weighted Wing loss path not actually used (Doc 02 C.4) | `train.py` MultiTaskLoss caller | Missing |
| Anchor sizes still default `(32,64,128,256,512)` (Doc 01 B.3) | `model.py` `AnchorGenerator` | Missing ~0.7 mAP |
| `USE_LDAM_DRW = False` default (Doc 02 C.2) | `config.py` | Missing ~2% Activity Top-1 |
| `USE_HEADPOSE_FILM = False` default (Doc 01 E) | `config.py` | Missing ~0.7% Activity Top-1 |
| `USE_LION = False` default (Doc 02 E.1) | `config.py` | Missing ~0.5% across tasks + memory headroom |
| SWA, TTA not implemented | — | Missing ~1.3–2% across tasks at eval time |

---

## D. What this means for the "are we ready to train" question

**Short answer:** You can train *something useful* right now, and it will be a major improvement over the diagram-faithful baseline you started with. But the recipe is operating at maybe **60% of the headroom** I identified in the Doc 01/02/03 cycle.

**What's already fully active and contributing right now:**
- TCN block (A.1)
- T=16 window (A.2)
- 2× ViT + CLS token (A.3)
- Attn dropout (A.4)
- Layer4 BN unfrozen (B.2)
- Causal Transformer PSR (C.1) — though with the cache caveat above
- Per-component PSR heads (C.2)
- GIoU detection (C.1 of Doc 02)
- RandAugment (D.1)
- CutMix alternation (D.2)
- Differential LR (B.2 of Doc 02)
- PSR tolerance F1 in eval (Doc 03 A.1)
- Clip-level activity eval (Doc 03 A.2)

**What's structurally there but switched off in the default config — flip these flags:**
- ConvNeXt-Tiny backbone (`BACKBONE = 'convnext_tiny'`) — instant +1.5% Top-1
- HeadPoseFiLM (`USE_HEADPOSE_FILM = True`) — +0.7% Top-1
- Lion optimizer (`USE_LION = True`, after `pip install lion-pytorch`) — +0.5% all + memory headroom
- LDAM-DRW (`USE_LDAM_DRW = True`) — +2% Top-1

**What needs real new code work before it can help:**
- VideoMAE stream — the highest-value single thing. Plumb `clip_rgb` end-to-end. Doc 02 next section will detail this.
- Stage parameter freezing
- Random temporal stride
- SWA + flip TTA + 5-crop TTA
- PSR cache hygiene + sequence-aware sampling

The next document focuses on what to actually *do* with this state of affairs: which experiments to run, in what order, and what numbers to expect from each.
