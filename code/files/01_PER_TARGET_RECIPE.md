# 01 — Per-Target Recipe to Beat Every IndustReal Benchmark

The MASTER_BUG_REPORT confirms 10 implementation bugs have been fixed. Implementation is now correct, but **correct ≠ winning**. To clear every target in `popwbenchmark.tex` and `popw_paper.tex` you still need the targeted upgrades below — they are not bug fixes, they are competitive headroom that the corrected baseline doesn't have.

This doc walks each headline target individually: what's currently in your code, why that won't clear the bar, and the specific intervention that does. Estimated probability of clearing each target after applying these is at the bottom.

The IKEA-ASM targets in the LaTeX papers are intentionally deferred — there is no IKEA-ASM dataset module in your codebase yet (`grep` confirms only `industreal_dataset.py` exists). Doc 03 covers the paper-level handling of that mismatch. This doc is IndustReal-only.

---

## A. ASD Detection mAP@0.5 — target > 83.80% (YOLOv8m COCO+synth+real)

### What you have today (post-bug-fix)

After Bug #0 is fixed, your `_encode_boxes` produces correct regression targets. The detection head is RetinaNet-style (4× Conv 3×3, focal α=0.25 γ=2, GIoU regression), 24 ASD classes, anchor sizes calibrated by k-means to `(24, 48, 96, 192, 384)`. Backbone is ConvNeXt-Tiny ImageNet-pretrained, layer1–3 frozen in Stage 1, BN unfrozen in layer4.

### Why this won't clear 83.80% by itself

YOLOv8m's 83.80% is **not** ImageNet-pretrained — the IndustReal paper trains it on COCO + synthetic IndustReal + real IndustReal. That is roughly 2.5× more detection-relevant pretraining data than POPW's ImageNet weights provide. Multi-task gradient interference makes this gap harder, not easier, even with Kendall weighting. Without targeted detection pretraining you should expect 80–82% mAP@0.5, just below the bar.

### The intervention

You already have `pretrain_synthetic.py` written. **It is not run by default** — `PRETRAIN_DET_ON_SYNTH = True` in config is a hint, not an action. The actual pretraining is a separate process you must launch before main training.

1. **Run `pretrain_synthetic.py`** for 20 epochs on the IndustReal training split with detection-only loss. The script already exists, freezes everything except backbone+FPN+detection head, and uses `lr=5e-4`. Expected mAP@0.5 on val after 20 epochs: 75–82%.

2. **Load the resulting checkpoint as initialization** for the main multi-task run. `train.py` accepts `--resume` and uses `strict=False` loading, so the missing heads (head pose, activity, PSR, FiLM modules) won't cause errors.

3. **Add box-level synthetic data if the IndustReal authors' synthetic frames are available.** The 83.80% number specifically uses COCO+synth+real. Their synthetic split is published on their GitHub; you can pretrain on it for an additional 10 epochs before fine-tuning on real IndustReal.

4. **Anchor calibration is already done** — `(24, 48, 96, 192, 384)` from k-means on the training boxes. No further work.

5. **NMS at IoU=0.5** during eval (already set as `DET_EVAL_NMS_IOU_THRESH=0.5`).

### Expected outcome with intervention

mAP@0.5: **84–87%**, with margin. Confidence after fixes: **75%**.

The specific failure mode to watch for: if pretrain mAP@0.5 lands below 75%, the ImageNet pretraining of ConvNeXt-Tiny is fighting the assembly distribution. In that case, pretrain for 30 epochs instead of 20, and consider using `MAE` pretraining via `pretrain_mae.py` first to adapt the visual statistics.

---

## B. Activity Top-1 — target > 66.45% (MViTv2 Kinetics-400)

### What you have today

ConvNeXt-Tiny + FPN, then activity head with: `concat(f_det[24] || GAP(C5_mod2)[768] || GAP(P4)[256])` → 1048→512 projection → feature bank T=16 → TCN → 2× ViT (8 heads, d_k=64) → CLS readout → LDAM-DRW classifier (74 cls, label_smooth=0.1). Random temporal stride {2,3,4,5} per clip enabled.

### Why this won't clear 66.45% by itself

The MViTv2 baseline is pretrained on Kinetics-400 (~250k clips). POPW starts from ImageNet still images. Kinetics has motion priors that ImageNet does not. Your TCN+ViT temporal modeling is sound, but it has no source of strong temporal pretraining. T=16 with stride 3 covers 1.6s context, which is the median action length — it's enough for short actions but won't help on long, multi-step actions like `screw` (5+ seconds).

The LDAM-DRW Bug #1 fix does help — margins now use raw counts, not effective-number weights, so rare classes get larger margins. This alone is +0.5 to +1.0% on macro-F1 but doesn't move Top-1 by much because Top-1 is dominated by frequent classes.

### The intervention

Three layered moves, ranked by leverage.

**B.1 (highest leverage) — Enable VideoMAE V2 stream.** `USE_VIDEOMAE = False` is the current default, but the entire infrastructure is wired:
- `VideoMAEStream` class in `model.py` (line 677)
- `clip_rgb` field in dataset output (`industreal_dataset.py` line 532)
- VideoMAE projection branch in `ActivityHead` (line 1019, gated on `use_videomae`)
- Collate function passes `clip_rgb` through to model

To enable: `USE_VIDEOMAE = True` in `config.py`, and `pip install transformers`. The stream takes 16-frame 224×224 clips, runs them through a frozen VideoMAE-Small (Kinetics-400 fine-tuned) checkpoint, projects 384-D output to 512-D, and concatenates with the CLS token output before the classifier. Memory cost: +22M frozen params, ~600 MB VRAM. Inference FPS drops ~25%, but accuracy gain is **+5 to +7% Top-1**, which is the difference between losing and decisively winning this benchmark.

**Important:** when you flip `USE_VIDEOMAE = True`, the activity classifier's input dim doubles (`embed_dim * 2 = 1024`). The `videomae_proj` projection is built at construction time. If you have a checkpoint trained with `use_videomae=False` and try to resume with `use_videomae=True`, the activity classifier shapes mismatch. Either retrain from scratch or load with `strict=False` and accept that the classifier head reinitializes.

**B.2 — Bring `LDAM_DRW_EPOCH` forward.** Current value is 60 (out of 100 total epochs). Stage 3 starts at epoch 16, so DRW-deferred reweighting only kicks in after activity training has been stable for 44 epochs. This is too late — the LDAM paper recommends DRW activate at "feature stability", which on IndustReal you should reach by epoch 30–35 (when your Stage 3 has had 15–20 epochs to stabilize). Set `LDAM_DRW_EPOCH = 35`. This is a single-line config change worth +0.5 to +1.0% Top-1 on long-tail classes.

**B.3 — Clip-level activity evaluation, not frame-level.** The IndustReal paper evaluates Top-1 on **action clips**, not single frames. Each clip has a single label and is sampled into 16 frames at uniform stride; the model's prediction is the argmax over the clip's 16 frame logits (or, better, the clip-level CLS readout). Your `evaluate.py` already computes a `_compute_clip_level_accuracy()` metric — make sure your reported `act_top1` headline number uses that, not per-frame accuracy.

If you currently report frame-level Top-1, switching to clip-level typically adds **+1.5 to +3% Top-1** because the model has 16 chances to land the correct class.

### Expected outcome with intervention

Top-1: **70–75%** with VideoMAE on, **66–69%** without. Confidence with VideoMAE: **80%**. Without VideoMAE: 55% — uncomfortable margin.

**Strong recommendation: turn VideoMAE on.** It is the single intervention that takes you from "competitive" to "decisive" on Activity. The VRAM hit fits on a 12 GB card if you keep `BATCH_SIZE=2` and accumulate.

---

## C. Activity Top-5 — target > 88.43% (MViTv2 Kinetics-400)

The easiest of the headline targets. With a competent 74-class classifier, Top-5 is forgiving.

### What you have today

Same activity head as above. Top-5 is mechanically derived from the logits.

### Expected outcome with intervention

Top-5 follows Top-1 by a roughly fixed margin on this kind of task — typically Top-5 is 18–22% above Top-1. So:
- Without VideoMAE (Top-1 ≈ 67%): Top-5 ≈ **87–90%**, marginal at the bar.
- With VideoMAE (Top-1 ≈ 73%): Top-5 ≈ **92–95%**, decisive.

Confidence with VideoMAE: **90%**. Without: **65%**.

No additional intervention beyond the Top-1 work above.

---

## D. PSR F1 — target > 0.901 (STORM-PSR, ±3 frame tolerance)

### What you have today

Multi-scale `concat(GAP(P3) || GAP(P4) || GAP(P5))` (768-D) → per-frame MLP → causal Transformer (3 layers, 4 heads, d=128) → 11 per-component output heads → 11-D logits, one per assembly component. Loss: binary focal (α=0.25, γ=2.0) + temporal smoothness (w=0.05). Cache for inference is bounded to `_MAX_CACHE_LEN=32`.

### Why this won't clear 0.901 by itself

This is the most architecturally serious gap. **At training time, PSR receives a single frame per sample.** Look at `industreal_dataset.py` line 504: `psr_labels = torch.from_numpy(cache.psr_per_frame[frame_num]).float()` — one frame, one set of labels. The Transformer processes T=1 with a 1×1 causal mask, which is no temporal modeling at all.

At evaluation time, the cache fills with up to 32 ordered frames and the causal mask works. But the model has been trained as a per-frame binary classifier with fancy temporal infrastructure that was never engaged. The Transformer's positional encoding, cross-frame attention, and KV-cache are all dormant during training. STORM-PSR was trained on full sequences with proper temporal context.

Per the IndustReal paper, the strongest baseline (B3 rule-based confidence aggregation) achieves F1=0.883. Your per-frame model can plausibly approach that ballpark because the underlying detection signal is there, but the leap from 0.883 to 0.901 (+1.8 F1 points) is exactly the temporal modeling contribution — and you don't have that during training.

### The intervention — sequence-mode PSR training

This is the single biggest unlock for PSR. It is implementation work, not a flag flip, but it transforms a 25%-confidence target into a 75%-confidence one.

**D.1 — Add a sequence-batched dataloader path.** In `industreal_dataset.py`, add an alternate `__getitem__` mode that returns a contiguous T-frame window from one recording with all 11×T PSR labels. The infrastructure is partially there (the dataset already iterates per-frame within recordings) but the collate function needs to handle variable-T or fixed-T batches.

Suggested layout:
- New dataset flag `sequence_mode: bool` and `sequence_length: int = 32`
- `__getitem__(idx)` in sequence mode returns `images: [T, 3, 720, 1280]`, `psr_labels: [T, 11]`, with all T frames coming from the same recording at consecutive sampling indices
- Collate stacks to `[B, T, 3, 720, 1280]` and `[B, T, 11]`

This is ~1.5 days of dataloader work. The shapes propagate cleanly through the model: the backbone is already shape-agnostic, and `PSRHead.forward` already supports a `[B, T, ...]` non-cached path (lines 1259–1272), it just needs T>1.

**D.2 — Adjust the training loop to alternate or co-batch.** The cleanest approach is **alternate-batch**: even-numbered training steps use random-frame mode (current behavior, drives detection/pose/activity), odd-numbered steps use sequence mode (drives PSR). You compute losses for the active task subset on each batch and use Kendall weighting to balance. This requires a small change in `train.py` to maintain two iterators and switch between them.

A simpler alternative: **PSR-only fine-tune**. Train normally for the full schedule (Stage 1–3), then add a `Stage 4` where you load the best checkpoint and fine-tune for 10–15 more epochs in sequence-mode using only PSR loss. This is less elegant but simpler to implement (~half day) and captures most of the gain.

**D.3 — Tolerance reporting.** STORM-PSR's 0.901 F1 is at ±3 frames. The IndustReal paper's B3 baseline (0.883 F1) is at ±5. Your `evaluate.py` already supports a `tolerance_frames` parameter. **Always report both** — at ±3 you compare against STORM-PSR, at ±5 against B3. A model that clears 0.901 at ±3 will be at ~0.92 at ±5; a model that clears 0.883 at ±5 may not clear 0.901 at ±3. The two are different targets.

**D.4 — Per-component DRW**. Component 0 (base plate) appears in 95% of "assembled" frames; component 10 (wheels) appears in <30%. The per-component output heads (already in code) help, but you should also weight the binary focal loss per-component using effective-number weights. This is a small `losses.py` change — pre-compute per-component prevalence from `cache.psr_per_frame.mean(axis=0)` and pass to the loss.

### Expected outcome with intervention

- With sequence-mode training (D.1+D.2): **F1 = 0.91–0.93 at ±3**, **0.92–0.94 at ±5**. Decisive.
- Without sequence-mode but with D.3+D.4 only: **F1 = 0.86–0.89 at ±3**, **0.89–0.92 at ±5**. Beats B3 at ±5, marginal vs STORM-PSR at ±3.

Confidence with full intervention: **75%**. With minimal intervention: **30%**.

The sequence-mode training is the single most expensive thing in this entire roadmap (~1.5 days of careful dataloader work) but it is non-optional if you want to claim a STORM-PSR win. Without it, the honest framing is that you beat B3 (the original IndustReal paper's baseline) and are competitive-but-not-better-than the follow-up STORM-PSR paper.

---

## E. PSR POS — target > 0.812 (STORM-PSR)

POS = Procedure Order Similarity (Damerau-Levenshtein-based, see Section 3.2 of the IndustReal paper). It measures whether the **ordered sequence** of step completions is correct end-to-end. One wrong step early in the sequence can cascade.

### What you have today

POS is a function of your per-frame component predictions, post-processed into a sequence of step completions. Your `evaluate.py` computes it from PSR logits.

### Why this won't clear 0.812 by itself

POS is dominated by the same problem as F1: per-frame training. With T=1 training, the model has no learned notion of step ordering. It can still produce a roughly correct sequence because component completions are temporally local (component i tends to be completed before component i+1 in IndustReal's procedures), but the model learns this only implicitly through the data distribution, not through the temporal structure.

### Expected outcome with intervention

- With sequence-mode training (D.1+D.2): **POS = 0.83–0.86**.
- Without: **POS = 0.74–0.79**, below the bar.

Confidence with full intervention: **70%**. Without: **20%**.

**There is no separate intervention for POS** — it rides on PSR F1's coattails. If you fix sequence-mode training for D, POS comes along.

---

## F. Head Pose 9-DoF — establish baseline

There is no published supervised baseline. This is an *establishing* task, not a beating task.

### What you have today

`HeadPoseHead`: `concat(GAP(C4) || GAP(C5))` (1152-D) → MLP(1152→512→256→9) with LayerNorm + GELU + Dropout. Loss: MSE × 0.001. Output: forward[3] || position[3] || up[3].

### Concerns

The reported headline metric should be **angular error in degrees** for forward and up vectors, plus **mm position error**. Your `evaluate.py` already normalizes the predicted vectors before computing the dot product (per prior fix), so `head_pose_angular_MAE_deg` is honest.

### Expected outcome

Forward angular MAE: **5–9°**. Up angular MAE: **4–7°**. Position MAE: **30–50 mm**.

These numbers are presentable as a baseline. The HeadPoseFiLM second-stage modulation depends on these predictions — bad head pose hurts activity, good head pose helps. Since `TRAIN_HEAD_POSE = False` (correctly, because IndustReal has no body keypoints), the only pose-related loss is the head pose MSE.

Confidence: **95%**. There's no explicit benchmark to clear; the work is in *defending* the numbers as reasonable.

---

## G. Assembly State F1@1 — target > ~0.85 (SupCon+ISIL ResNet-34)

Derived from your detection head. F1@1 = "for each frame with a single annotated state, did the top-confidence detection match?"

### What you have today

24-class detection head; F1@1 follows directly from detection mAP and per-class behavior.

### Expected outcome

If your ASD mAP@0.5 lands at 84–87% (Section A above), F1@1 will be **0.83–0.88**. The relationship is: F1@1 typically tracks mAP@0.5 within ±2 points on this dataset, because most frames have one dominant state and the top-confidence detection is usually correct when mAP is high.

Confidence: **70%** if Section A succeeds, **35%** if it doesn't.

No additional intervention beyond Section A.

---

## H. Error Verification AP — target > ~0.58 (GCA model)

Derived from confidence margin between detected state and expected state.

### What you have today

The detection head outputs class scores; the expected state at each frame is known from the procedure annotation. Error verification score = `1 - confidence(expected_state)`. Your `evaluate.py` should be computing AP from these scores; if it's not, add it.

### Expected outcome

AP **0.62–0.68** if detection is healthy. The GCA baseline at 0.58 used only ResNet-34 contrastive learning; your end-to-end multi-task model with 84%+ mAP detection should clear it comfortably.

Confidence: **70%** if Section A succeeds, **35%** if it doesn't.

No additional intervention beyond Section A.

---

## I. Probability summary table

| Target | Strongest baseline | Probability if Section A only | Probability with full Doc 01 plan | Required intervention |
|---|---|---|---|---|
| ASD mAP@0.5 > 83.80% | YOLOv8m | 75% | 75% | Section A: synthetic pretraining |
| Activity Top-1 > 66.45% | MViTv2 | 55% | **80%** | Section B: VideoMAE on, DRW=35, clip-level eval |
| Activity Top-5 > 88.43% | MViTv2 | 65% | **90%** | Same as Top-1 |
| PSR F1 > 0.901 (±3) | STORM-PSR | 30% | **75%** | Section D: sequence-mode PSR training |
| PSR F1 > 0.883 (±5) | B3 baseline | 80% | **90%** | Section D: minimum tolerance reporting |
| PSR POS > 0.812 (±3) | STORM-PSR | 20% | **70%** | Section D: sequence-mode PSR training |
| Assembly State F1@1 | SupCon+ISIL | 70% | 75% | Riders on Section A |
| Error Verification AP | GCA | 70% | 75% | Riders on Section A |
| Head Pose baseline | (no baseline) | 95% | 95% | Honest reporting only |

**Expected outcome of the full Doc 01 plan: clears 8 of 8 IndustReal headline benchmarks, with comfortable margin on 5 of them.** The two thinnest margins are PSR F1 at ±3 (75%) and ASD mAP@0.5 (75%) — both depend on a single critical intervention (sequence-mode PSR for the first, synthetic pretraining for the second).

The two interventions you cannot skip:
1. **Run `pretrain_synthetic.py` before main training.** Cost: 8 hours overnight on RTX 3060. Skipping costs ~3 mAP@0.5.
2. **Implement sequence-mode PSR training.** Cost: 1.5 days. Skipping concedes the STORM-PSR comparison.

Everything else in this doc (VideoMAE, DRW=35, anchor calibration, clip-level eval, per-component DRW) is flag-flipping or one-line edits.

The remaining strategic and engineering work (training schedule, monitoring, paper framing) is in Doc 02 and Doc 03.
