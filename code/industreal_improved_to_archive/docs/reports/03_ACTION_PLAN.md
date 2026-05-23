# 03 — Action Plan: From Current State to Publishable Results

This is the concrete sequence of things to do. Each step has a time estimate, a measurable outcome, and a decision point about whether to continue or pivot.

The plan is organized so that **after each step you have a defensible result** — not "do everything and pray." If thesis time pressure forces you to stop early, you'll still have something publishable.

---

## Phase 0 — Pre-flight (1 day)

Before any training, fix the showstopper bugs from Audit Doc 01.

### 0.1 Fix the PSR cache leak (MUST DO)

`model.py` `PSRHead._cache` grows unbounded. Add this in `forward()`:

```python
# Limit cache size to prevent memory leak
MAX_CACHE_LEN = 32  # ~3 seconds at 10 FPS, enough context for PSR
if len(self._cache[key]) > MAX_CACHE_LEN:
    self._cache[key] = self._cache[key][-MAX_CACHE_LEN:]
```

Without this, your training will OOM by epoch 5–10 of long runs.

### 0.2 Decide on PSR training mode (MUST DO)

The current PSR cache assumes ordered frames. With shuffled training batches, this is broken. Two options:

**Option A — quickest fix:** disable the cache during training, only use it at inference.
```python
# In PSRHead.forward, gate the cache path on self.training:
should_cache = (video_ids is not None and camera_views is not None) and not self.training
```
This makes training equivalent to per-frame PSR (which still uses multi-scale features and per-component heads — better than the original BiGRU baseline). Loses ~0.02–0.04 F1 vs proper sequence training but is safe.

**Option B — proper fix:** add sequence-mode dataloading. See `industreal_dataset.py` — modify it to optionally yield contiguous frame sequences of length T. At ~16 frames per "batch element", you can process them all at once and the causal mask works. This is ~1 day of dataloader work.

**My recommendation:** Start with Option A for the first training run. If PSR F1 lands in the 0.86–0.89 range, you're already competitive with the B3 rule-based baseline (0.883). If you need to push higher to clear STORM-PSR (0.901), do Option B as a second iteration.

### 0.3 Smoke test (MUST DO)

```bash
# Run a single epoch with debug mode to make sure nothing crashes
DEBUG_MODE=True python train.py --max-epochs 1
```

Watch for:
- NaN losses in the first 100 steps → indicates a loss scaling problem
- OOM at any point → reduce batch size or grad_accum_steps
- "shape mismatch" errors → indicates VideoMAE-related bug from Audit Doc 01 §B.1 (should not trigger if `USE_VIDEOMAE=False`, which is the default)

### 0.4 Establish a baseline reference number (1–2 days)

**Run the default config end-to-end on a debug subset (~20 recordings, 20 epochs) before any flag flipping.** This gives you:
- A "before" number to measure improvements against
- A sanity check that the eval pipeline works
- Realistic training time estimates

Expected outcome: ~30–45 min/epoch on RTX 3060 in debug mode. ~20 epochs = 10–15 hours overnight. The numbers won't beat baselines but they'll tell you the model trains stably and the eval pipeline reports plausible values.

**Decision point after Phase 0:** if the baseline run lands within ±3% of expectations (Top-1 ~58%, mAP ~75%, PSR F1 ~0.80), continue. If it's wildly off (Top-1 below 50%, NaN losses, etc), debug before scaling up.

---

## Phase 1 — Easy wins (3–5 days, including training time)

This is where you flip 4 flags and run synthetic pretraining. Each item is gated so you can stop after any of them with a defensible result.

### 1.1 Run synthetic detection pretraining (1 day + overnight training)

```bash
python pretrain_synthetic.py  # 20 epochs on detection only
# saves to runs/pretrain_synthetic/checkpoints/best.pth
```

Expected outcome: detection-only mAP@0.5 around 75–82% on val after 20 epochs. This is your strong starting point for the multi-task run.

### 1.2 Flip the 4 config flags

Edit `config.py`:
```python
BACKBONE = 'convnext_tiny'         # +1.5% Top-1, slightly faster
USE_HEADPOSE_FILM = True            # +0.7% Top-1
USE_LDAM_DRW = True                 # +2% Top-1 (long-tail fix)
USE_LION = True                     # +0.5% all + memory headroom
```

Then:
```bash
pip install lion-pytorch
```

### 1.3 Main training run with pretrained detection checkpoint (3–5 days)

Modify `train.py` invocation to load the synthetic-pretrained checkpoint:

```bash
python train.py --resume runs/pretrain_synthetic/checkpoints/best.pth --max-epochs 60
```

Note: the existing resume code may need a small tweak — it expects a full checkpoint dict, the pretrain script saves only `{'model': ...}`. Add a `strict=False` or skip-non-matching-keys path in the loader.

Training time on RTX 3060: ~1.5 hours/epoch × 60 epochs = ~90 hours = 4 days. With early stopping (patience=10), more like 30–40 epochs = 2.5 days.

### 1.4 Evaluate and decision point

Run `evaluate.py` on the test split. Expected results in this config:

| Target | Expected | Likely outcome |
|---|---|---|
| ASD mAP@0.5 | 83–85% | **clears 83.8% target marginally** |
| Activity Top-1 | 67–70% | **clears 66.45% target with margin** |
| Activity Top-5 | 90–92% | **clears 88.43% comfortably** |
| PSR F1 (tolerance=3) | 0.86–0.89 | **clears 0.883 (B3) but below 0.901 (STORM-PSR)** |
| PSR POS | 0.78–0.81 | **misses 0.812 target slightly** |
| Head Pose forward MAE | 7–10° | establishes baseline |
| Assembly State F1@1 | 0.84–0.87 | **clears baseline** |
| Error Verification AP | 0.62–0.66 | **clears baseline** |

**This is a publishable result.** You beat ~5–6 of 8 targets including the headline ASD and Activity numbers. PSR misses STORM-PSR but beats B3.

**Decision point:**
- If you stop here, you have a paper that beats most IndustReal benchmarks. The story: "unified multi-task model with PoseFiLM + causal Transformer PSR beats specialized baselines on detection, activity, and assembly state recognition."
- If you have more time, continue to Phase 2 to also clear PSR and bump Activity Top-1 to a more decisive margin.

---

## Phase 2 — Critical code fixes (1 week)

These four items unlock the remaining gains. They're code work, not just flag-flipping.

### 2.1 PSR sequence-mode training (2 days)

Implement Option B from §0.2:
- Add a `SequenceMode` flag to `industreal_dataset.py`
- In sequence mode, `__getitem__` returns a window of T contiguous frames from one recording (with their per-frame labels)
- Modify `train.py` to alternate batches between random-frame mode (for det/pose/activity) and sequence mode (for PSR)
- Set `T_PSR = 32` (about 3 seconds, enough for procedure transitions)

Expected gain: **+0.03–0.05 PSR F1**, putting you at 0.91+ (clears STORM-PSR).

### 2.2 Wire VideoMAE V2 stream end-to-end (1–2 days)

Three pieces:

**Piece 1 — dataloader.** Add a `clip_rgb` field to dataset output:
```python
# In industreal_dataset.py __getitem__, add:
clip_indices = self._sample_clip_indices(idx, T=16)  # uniform 16 frames around current
clip_frames = [self._load_frame(i) for i in clip_indices]
clip_rgb = torch.stack([
    self._resize_to_videomae_input(f) for f in clip_frames  # → 224×224
])
return {'image': image, 'clip_rgb': clip_rgb, 'targets': ...}
```

**Piece 2 — training loop.** Pass `clip_rgb`:
```python
# In train.py train_one_epoch:
clip_rgb = batch['clip_rgb'].to(device) if C.USE_VIDEOMAE else None
outputs = model(images, video_ids=video_ids, clip_rgb=clip_rgb)
```

**Piece 3 — handle None gracefully in ActivityHead.** Currently `use_videomae=True` forces `classifier_input_dim = embed_dim * 2 = 1024`, but if `videomae_feat is None`, the classifier crashes on a 512-D input. Fix:
```python
# In ActivityHead.forward:
if self.use_videomae and videomae_feat is not None:
    videomae_emb = self.videomae_proj(videomae_feat)
    feat = torch.cat([feat, videomae_emb], dim=-1)
elif self.use_videomae:
    # Pad with zeros for missing VideoMAE feature (shouldn't happen in practice)
    feat = torch.cat([feat, torch.zeros_like(feat)], dim=-1)
```

Also: the first 10 epochs the VideoMAE encoder is frozen (per Doc 02 A.1). After epoch 10, unfreeze with a tiny LR (1e-5) added to the optimizer. This is a 5-line change.

Expected gain: **+5–7% Activity Top-1**, putting you at 73–77%. This single change is the difference between a "competitive" paper and a "decisive" paper on Activity.

### 2.3 SWA + flip TTA + 5-crop TTA (1 day)

**SWA at end of training:**
```python
# At the end of train.py main():
if C.USE_SWA:
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    for epoch in range(C.EPOCHS, C.EPOCHS + 8):
        train_one_epoch(model, train_loader, optimizer, ...)
        swa_model.update_parameters(model)
        swa_scheduler.step()
    update_bn(train_loader, swa_model)
    torch.save(swa_model.state_dict(), checkpoint_dir / 'swa_final.pth')
```

**Flip TTA in evaluate.py:**
```python
def predict_with_tta(model, image, do_flip=True):
    out1 = model(image)
    if not do_flip:
        return out1
    out2 = model(torch.flip(image, dims=[3]))
    # Flip detection boxes back
    out2['cls_preds'] = ...  # left-right swap class indices for symmetric classes
    # For activity: just average logits
    avg_act = 0.5 * (out1['act_logits'] + out2['act_logits'])
    return {**out1, 'act_logits': avg_act}
```

**5-crop TTA for activity at clip-level evaluation:**
```python
# 4 corners + center, average logits
crops = [center_crop, tl_crop, tr_crop, bl_crop, br_crop]
all_logits = torch.stack([model(c)['act_logits'] for c in crops])
avg_logits = all_logits.mean(dim=0)
```

Expected gain: **+1.3–2% across all classification metrics**, applied at evaluation time only.

### 2.4 Stage parameter freezing (half day)

```python
# Add to train.py:
def _set_stage_requires_grad(model, stage):
    """Stage 1: layer1-3 frozen. Stage 2: + activity/PSR heads frozen. Stage 3: all trainable."""
    if stage == 1:
        for name, p in model.named_parameters():
            if any(ln in name for ln in ['layer1', 'layer2', 'layer3']):
                p.requires_grad = False
            elif 'activity_head' in name or 'psr_head' in name:
                p.requires_grad = False
    elif stage == 2:
        for name, p in model.named_parameters():
            if any(ln in name for ln in ['layer1', 'layer2']):  # only layer1-2 frozen now
                p.requires_grad = False
            elif 'activity_head' in name or 'psr_head' in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
    else:  # stage 3
        for p in model.parameters():
            p.requires_grad = True

# In train_one_epoch, at the start:
prev_stage = getattr(model, '_current_stage', 0)
if stage != prev_stage:
    _set_stage_requires_grad(model, stage)
    # Rebuild optimizer to drop frozen params (or use param groups with lr=0)
    model._current_stage = stage
```

Expected gain: **+0.5–1% across most metrics**, faster convergence.

### 2.5 Re-evaluate after Phase 2

Expected results after Phase 1 + Phase 2:

| Target | Phase 1 result | + Phase 2 result | vs target |
|---|---|---|---|
| ASD mAP@0.5 | 83–85% | 86–88% | comfortable margin |
| Activity Top-1 | 67–70% | 73–77% | decisive |
| Activity Top-5 | 90–92% | 92–94% | decisive |
| PSR F1 (tolerance=3) | 0.86–0.89 | 0.91–0.93 | clears STORM-PSR |
| PSR POS (tolerance=3) | 0.78–0.81 | 0.83–0.86 | clears STORM-PSR |
| Head Pose forward MAE | 7–10° | 6–8° | tight baseline |
| Assembly State F1@1 | 0.84–0.87 | 0.87–0.90 | comfortable |
| Error Verification AP | 0.62–0.66 | 0.65–0.70 | comfortable |

This is a "beat all major baselines" result.

---

## Phase 3 — Publication rigor (1 week)

You have results. Now make them defensible.

### 3.1 Multi-seed runs (3–5 days)

Re-run the final config with seeds 42, 123, 7. Report mean ± std for headline numbers. This is non-negotiable for any reviewer.

```bash
for SEED in 42 123 7; do
    python train.py --seed $SEED --output_dir runs/seed_$SEED
    python evaluate.py --checkpoint runs/seed_$SEED/checkpoints/best.pth
done
```

A 2.5% improvement with std 0.4% is publishable. With std 1.8%, it's noise. Your training loss curves should give you an early indication — if losses are smooth and converge to similar values across seeds, you're fine.

### 3.2 Ablation table (1 day)

Plan the full ablation table before running:

| Config | ASD mAP | Top-1 | PSR F1 |
|---|---|---|---|
| Baseline (XML diagram, T=8, 1×ViT, BiGRU PSR) | 78.0 | 65.0 | 0.85 |
| + TCN | 78.0 | 66.4 | 0.85 |
| + T=16 | 78.0 | 67.3 | 0.85 |
| + 2× ViT + CLS | 78.0 | 68.9 | 0.85 |
| + Causal Transformer PSR | 78.0 | 68.9 | 0.88 |
| + Per-component PSR heads | 78.0 | 68.9 | 0.89 |
| + GIoU + layer4 BN | 81.5 | 69.1 | 0.89 |
| + Synthetic pretraining | 84.2 | 69.5 | 0.90 |
| + ConvNeXt-Tiny | 84.5 | 71.0 | 0.90 |
| + HeadPoseFiLM | 84.5 | 71.7 | 0.90 |
| + LDAM-DRW + Lion | 84.7 | 73.5 | 0.90 |
| + VideoMAE stream | 84.7 | 76.5 | 0.90 |
| + Sequence-mode PSR | 84.7 | 76.5 | 0.92 |
| + SWA + TTA | 86.5 | 78.0 | 0.92 |
| **POPW-Full** | **86.5** | **78.0** | **0.92** |

You don't need 14 ablations — a clean ~6-row table covering the major architectural categories is enough. Pick the rows where the contribution is clearest:
1. Baseline (per XML diagram)
2. + Architecture changes (TCN, T=16, 2× ViT, causal PSR — all together)
3. + Loss & training changes (GIoU, LDAM, layer4 BN, RandAugment, CutMix)
4. + Pretraining (synthetic)
5. + Backbone (ConvNeXt-Tiny + HeadPoseFiLM)
6. + VideoMAE
7. **POPW-Full**

For each row, train for ~30 epochs (faster than the 60-epoch full runs) with seed=42 only. Total time: ~6 × 1.5 days = 9 days. **Plan this in advance because it's the longest single time sink.**

### 3.3 Per-class breakdown (half day)

For Activity (74 classes) and ASD (24 classes), report:
- Per-class F1 sorted ascending
- Top-5 hardest classes with their F1
- Confusion matrix figure (heatmap-style)

This pre-empts "did you cheat by getting only easy classes?" Your `evaluate.py` already computes per-class F1 (I saw `f1_score(..., average=None)` patterns). Just dump them to a CSV and plot the bottom-5/top-5.

### 3.4 Efficiency reporting (half day)

Run `efficiency_report.py --all_configs --onnx_export` and put the output table in your thesis. Add:
- Streaming FPS measurement (single-frame update with cached temporal bank)
- Multi-model baseline equivalent comparison ("YOLOv8m + MViTv2 + STORM-PSR sequential" vs POPW)

### 3.5 Qualitative figures (1 day)

For impact:
- A 3-row figure showing one frame per major task: detection bboxes overlay, head pose forward-vector arrow, PSR component states bar
- A timeline plot showing PSR component state evolution over a recording vs ground truth
- Activity confusion matrix heatmap

These are the figures reviewers remember.

---

## Phase 4 — Stop conditions

Don't keep iterating past these points:

| Stop condition | Why |
|---|---|
| You've hit Phase 1 results and have <2 weeks left | Phase 1 alone is publishable. Don't risk it for marginal Phase 2 gains |
| ASD mAP@0.5 ≥ 84%, Activity Top-1 ≥ 70%, PSR F1 ≥ 0.89 | You've cleared all major IndustReal targets; further work has diminishing returns |
| Phase 2 doesn't move the needle by epoch 20 | Either there's a bug or you've hit the model's ceiling. Spend the time on writing/figures |
| You're 6+ months in and haven't run multi-seed | Stop adding features, just run seeds 123 and 7 with what you have |

---

## Phase 5 — What to write in your thesis

The framing that makes POPW look strongest:

**Don't claim:** "POPW is the most efficient multi-task model" (PTMA wins)
**Don't claim:** "POPW beats all single-task baselines on every metric" (some specialized models will edge you out somewhere)

**Do claim:**
1. **Unified multi-task architecture** — one forward pass produces detection, pose, head pose, activity, and PSR outputs. The 5 tasks share a common backbone with PoseFiLM cross-task conditioning.
2. **Per-task competitive with specialized models** — beats YOLOv8m on ASD (with appropriate caveats about pretraining data), competitive with MViTv2 on Activity (with VideoMAE pretraining), beats STORM-PSR on PSR (with sequence training).
3. **Streaming-capable** — feature-bank design lets POPW process new frames incrementally. Cite a streaming FPS number that's competitive with batch FPS of single-task competitors.
4. **Architectural novelty** — PoseFiLM + HeadPoseFiLM cross-task conditioning is genuinely new for industrial assembly recognition.

---

## Bottom line — the realistic timeline

| Path | Time | Outcome |
|---|---|---|
| Stop after Phase 0 (debug only) | 1 day | Confirms training works; not publishable |
| Stop after Phase 1 (easy wins) | ~1.5 weeks | Beats 5/8 IndustReal targets; **publishable, defensible** |
| Stop after Phase 2 (critical fixes) | ~3 weeks | Beats 7–8/8 IndustReal targets; **strong publishable** |
| Through Phase 3 (rigor) | ~5 weeks | Multi-seed, ablations, figures; **thesis-grade** |

Your immediate next action is **Phase 0.1 (PSR cache fix) + Phase 0.2 (decide on Option A) + Phase 0.3 (smoke test) + Phase 0.4 (debug-mode baseline)**. That's 1 week of work that gets you to a solid starting point. Don't skip it for the temptation to immediately run Phase 1.
