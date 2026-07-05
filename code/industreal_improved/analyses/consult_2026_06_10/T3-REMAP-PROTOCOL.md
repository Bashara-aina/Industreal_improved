# T3: MViTv2-S 75→69-class Remap Protocol (Opus 118 §7.18 + 126 Decision 4)

**Date:** 2026-07-05
**Goal:** Make our per-frame activity metric SOTA-comparable to MViTv2-S (WACV 2024 Tab 2).

---

## The protocol mismatch (before T3)

| Aspect | Our model | MViTv2-S (WACV 2024 Tab 2) | Fair? |
|---|---|---|---|
| **Output classes** | 69 (verb-grouped, hybrid mode) | 75 (fine-grained) | ❌ Different |
| **Temporal context** | 1 frame (per-frame MLP) | 16 frames (video clip) | ❌ Different |
| **Pretraining** | None (from-scratch) | Kinetics-400 | ❌ Different |
| **Sampling** | Per-frame argmax | 16-frame clip majority vote | ❌ Different |

T3 remap only addresses the **output classes** (75→69). The other 3 differences are addressable separately (T2 for temporal, Q26 for pretrain).

---

## T3 protocol (sum probabilities, never average/max)

The fair remap rule (per Opus 118 §7.18 and 126 §4.3):

> MViTv2 outputs 75-class softmax → sum probabilities within each of 69 groups → argmax over 69.
> This is **exactly the same protocol** our per-frame inference uses (`compute_activity_metrics` already does this in `src/evaluation/evaluate.py:915+`).

**Why sum, not average/max:**
- **Sum** preserves total probability mass (P(group) = sum of P(components in group))
- **Average** underweights large groups (P(group) = mean of P(components))
- **Max** discards information (P(group) = max P(components) — one outlier dominates)

**Bit-identical sanity check:** For ungrouped classes (75 = 69 case, which is the inverse mapping), MViTv2's argmax should be identical before and after the group collapse.

---

## The remap table

Source: `src/runs/rf_stages/checkpoints/act_remap_75_to_69.json` (built from `ACT_CLASS_GROUPING=hybrid` config in `src/config.py`).

| Statistic | Value |
|---|---|
| Source | `src/config.py:ACT_CLASS_NAMES` |
| Grouping mode | `hybrid` (verb-group for tail, standalone for ≥100 frames) |
| Raw classes | 75 (MViTv2 WACV protocol) |
| Output groups | 69 (matches our per-frame model) |
| Group names | 'other', 'take_short_brace', 'align_objects', ..., 'browse_instruction', ..., 'fit_short_brace' |

The 75→69 mapping is **data-driven** (built from AR_labels.csv frame counts), not a hand-crafted lookup.

---

## The training recipe (T3 script: `src/training/train_t3_mvitv2.py`)

1. **Model:** MViTv2-S, 16-frame clips, 32-frame effective temporal receptive field (stride=2)
2. **Pretrain:** Kinetics-400 (frozen, fine-tune head only by default)
3. **Data:** remap 75→69 via sum-of-probabilities, per-frame softmax
4. **Loss:** cross-entropy on 69-class labels
5. **Optimizer:** AdamW lr=1e-4, wd=0.01, cosine schedule
6. **Epochs:** 25 (WACV protocol baseline)
7. **Batch size:** 8 (fits 16GB GPU with 720×1280 input)
8. **Output:** `src/runs/rf_stages/checkpoints/t3_mvitv2_act.pth` (encoder + remapped head)

---

## Expected results

Per Opus 118 hypothesis (Q45): **MViTv2-S remapped Top-1 ≈ 0.25-0.35** (down from 0.65 raw 75-class, because the remap introduces new "merged" classes that confuse argmax).

This is **the bar** for our per-frame MLP to compare against. If our `act_top1_raw` ≥ 0.30 on the same 69-class protocol, we have a fair comparison.

---

## BLOCKER: Pretrained weights not available

| Source | URL | Status |
|---|---|---|
| `dl.fbaipublicfiles.com` (original PyTorchVideo) | `MVIT_RGB_16x4.pth` | ❌ 403 dead |
| `huggingface.co/facebook/mvit-base-16x4-kinetics400` | API | ❌ 401 gated |
| GitHub `facebookresearch/pytorchvideo` | `model_zoo/` | ❌ 404 deleted |

**T3 is BLOCKED** on pretrained weights. Three options to unblock:

1. **Wait for HF to lift the gate** on `facebook/mvit-base-16x4-kinetics400`
2. **Use a different MViTv2 mirror** (e.g., torchvision `mvit_v2_s` has K400 weights at PyTorch.org)
3. **Use the existing `slowfast_r50` baseline** (the architecture code supports it; has public K400 weights)

**Recommendation:** Once D3-redo and YOLOv8m free the GPUs (~1h for D3, ~15h for YOLOv8m), use the available GPU time to **train T3 with `slowfast_r50` instead** (already a supported model in `video_stream.py`). This gives us a 75→69 baseline immediately.

---

## T3 work that's DONE (CPU-only, no GPU needed)

| Task | Status | File |
|---|---|---|
| Extract 75→69 remap table from config | ✅ DONE | `src/runs/rf_stages/checkpoints/act_remap_75_to_69.json` |
| Write T3 training script | ✅ DONE | `src/training/train_t3_mvitv2.py` |
| Write T3 protocol document | ✅ DONE (this file) | `analyses/consult_2026_06_10/T3-REMAP-PROTOCOL.md` |
| Pre-process remapped labels into clip dataset | ✅ DONE | `IndustRealActivityDatasetRemapped` in T3 script |
| Write fair-comparison evaluation | ✅ DONE | `remap_75_to_69()` function in T3 script |

## T3 work that's BLOCKED (GPU/weights required)

| Task | Status | Blocked by |
|---|---|---|
| Download MViTv2-S K400 weights | ❌ URL 403 | Public mirror dead |
| Run T3 training | ❌ Wait | GPU (D3-redo on 3060, YOLOv8m on 3060, main training on 5060 Ti) |
| Eval T3 vs our per-frame MLP on remapped 69-class | ❌ Wait | T3 training must complete |

**ETA for unblocking:** Once D3-redo finishes (~1h), 3060 has 1 free slot. T3 training can start on 3060 with `slowfast_r50` fallback (no MViTv2 weights needed). Estimated 1-2 days for T3 training + eval.
