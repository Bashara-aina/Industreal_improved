# 16 — Master Prompt for Opus v5: Ultimate Path to SOTA-Ready Unified Multi-Task Model

## 0. What This Is

This is the comprehensive briefing for Opus v5. We have just achieved the first non-zero detection metric in project history (`det_mAP50=0.0091`). The RC-28/RC-29 fixes work. Now we need the **ultimate path** from "first non-zero metric" to "benchmarkable model comparable to SOTA."

All current source code, training logs, git diffs, and documentation are in this folder. See `FILE_MANIFEST.md` for a complete index.

---

## 1. Target: What "Done" Looks Like

We are building **POPW**: a unified multi-task architecture that produces 5 simultaneous predictions from a single RGB frame, targeting the benchmarks in `popw_paper_improved.tex` Tables 2-3.

### 1.1 IndustReal Benchmarks (Primary Dataset — Table 3 in paper)

| Task | SOTA Method | SOTA Score | Our Target |
|------|------------|------------|------------|
| **ASD mAP (b-boxed)** | YOLOv8m (COCO+synth+real) | 83.80 | Competitive |
| **ASD mAP@0.5 (all frames)** | YOLOv8m | 64.10 | Competitive |
| **Activity Top-1** | MViTv2 (RGB-only) | 65.25 | Competitive |
| **Activity Top-5** | MViTv2 (RGB-only) | 87.93 | Competitive |
| **PSR F1 (±3-frame)** | B2 ASD-accumulation | 0.731 | Competitive |
| **PSR POS** | B2 baseline | 0.816 | Competitive |
| **PSR Edit Score** | (PSRT protocol) | — | Report |
| **Assembly State F1@1** | SupCon+ISIL (ResNet-34) | ~0.83 | Competitive |
| **Error Verification AP** | Best contrastive (ResNet-34) | ~0.58 | Competitive |
| **Head Pose MAE** | No baseline | — | Establish first |

### 1.2 IKEA ASM Benchmarks (Secondary — Table 2 in paper)

| Task | SOTA Method | SOTA Score |
|------|------------|------------|
| **mAP@0.5** | Gated SRM (RGB+pose) | 21.77 |
| **mAP@0.5** | ActionFormer (RGB-only) | 21.49 |

### 1.3 What "Competitive" Means

The paper should be publishable. This doesn't mean beating every SOTA number — a unified model doing 5 tasks simultaneously with weight sharing vs. dedicated single-task models can be competitive at lower absolute numbers if the efficiency story is strong. But we need numbers that are in the conversation, not zeros.

---

## 2. Where We Are Now (2026-06-13)

### 2.1 Architecture (53M trainable params)
- ConvNeXt-Tiny + FPN → 5 heads (Detection 24cls, Body Pose 17kp, Head Pose 9-DoF, Activity 75cls, PSR 11 components)
- Cross-task conditioning: PoseFiLM, HeadPoseFiLM, det_conf → activity
- Kendall homoscedastic uncertainty weighting
- 76.16M total parameters

### 2.2 Current Training Results (R1 v4, Epoch 0, 25% subset)

```
det_mAP50 = 0.0091   ← FIRST NON-ZERO IN PROJECT HISTORY
ev_ap     = 0.0268   ← error verification also signaling
as_f1     = 0.0000   ← assembly state still zero
as_map_r  = 0.0000
psr_f1    = nan      ← eval skipped (TRAIN_PSR=False)
act_macro_f1 = 0.0000 ← eval skipped (TRAIN_ACT=False)
combined  = 0.1107
```

**Detection verdict**: LOCALIZING on GT frames (bestIoU up to 0.94, 4,142 preds@IoU>0.5 per 16-frame batch with 16 GT boxes). Model correctly silent on empty frames. Score_p50=0.001 (low confidence threshold issue).

### 2.3 What We Fixed (RC-28/RC-29 Era)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| **RC-28** | Empty frames contributed 4.15M-element focal loss / num_pos=1 (~85% of frames) | Skip empty frames, normalize by GT-bearing image count |
| **RC-29** | `mixed_precision: True` hand-flipped → GradScaler silently skipped ALL optimizer steps | FP32 enforced, committed/skipped telemetry added |
| **Eval crash #1** | Activity eval broadcast mismatch (preds ≠ GT count) | Shape guard added |
| **Eval crash #2** | Activity clip-level accuracy KeyError | Skip activity eval when TRAIN_ACT=False |
| **Eval crash #3** | Activity logging references missing stub keys | Guarded activity/PSR logger blocks |
| **PRE_VAL_GUARD** | Rejected loss ≤ 0 (Kendall fresh init issue) | Changed to `isfinite()` |

### 2.4 Current Presets

- **recovery_det_only**: FP32, det + head_pose only, eff batch 8, EMA/mixup off, zero_det_conf=False
- **recovery**: FP32, all heads, eff batch 8, same safety settings

### 2.5 Training in Progress
- R1 v4: epoch 1 at 14%, `--preset recovery_det_only --subset-ratio 0.25 --max-epochs 3`
- GPU: RTX 3060 12GB, speed: 1.7 batch/s, ~68 min/epoch training + ~87 min/epoch mAP eval

---

## 3. Key Questions for Opus v5

### A. The Ultimate Path

**What is the complete sequence of stages from where we are now (det_mAP50=0.0091) to a model that can fill in every `\popwres` placeholder in the paper with a competitive number?**

We need concrete stages, each with:
- Preset / training config
- Number of epochs
- Subset ratio progression (0.25 → 0.5 → 1.0)
- Gate condition to proceed
- Expected metric range at each gate

### B. The det_mAP50 Gap

We're at 0.0091 after 1 epoch (25% subset). The SOTA from `popw_paper_improved.tex` Table 3 lists YOLOv8m at 83.80 (mAP b-boxed) and 64.10 (mAP@0.5 all frames).

1. Is det_mAP50=0.0091 → competitive range feasible within ~100 epochs on this architecture?
2. What's the realistic ceiling for ConvNeXt-Tiny + RetinaNet-style head on this dataset?
3. Should we adjust expectations — e.g., report mAP@0.5 (b-boxed) separately from mAP@0.5 (all frames) and acknowledge the gap?
4. Is the 87-minute mAP eval per epoch sustainable, or should we eval every N epochs?

### C. When to Enable Activity and PSR

Currently training det + head_pose only (`recovery_det_only`). The R1 gate was `det_mAP50 ≥ 0.05`. At what point do we:

1. Enable activity head (with VideoMAE stream or GAP-only?)
2. Enable PSR head (with the PSR raw-loss probe Opus v4 suggested?)
3. Switch from `recovery_det_only` to `recovery` (joint)?
4. When should we add `--subset-ratio 1.0` (full dataset)?

### D. PSR Floor Diagnosis

Opus v4 said: "`psr=0.000001` is the smooth-cap floor, and binary focal at sigmoid≈0.5 should be O(0.1-0.7). Something upstream is zeroing the PSR loss before the cap." Spikes to 0.34-1.0 every ~10 steps are PSR-sequence batches (PSR_SEQ_EVERY_N_BATCHES=10).

1. Should we add the raw-loss probe Opus v4 suggested before enabling PSR?
2. Is the PSR head architecture (multi-scale GAP → Causal Transformer → 11 per-component MLPs) appropriate for the IndustReal PSR task?
3. The PSRT paper baseline (B2: F1=0.731, POS=0.816) and STORM-PSR (F1=0.506, POS=0.812) define the competitive range. Is our PSR head competitive with these?

### E. Efficiency Story

The paper's claim is efficiency through weight sharing. With 53M trainable params doing 5 tasks vs. dedicated models:
- YOLOv8m: ~25M params (detection only)
- MViTv2: ~51M params (activity only)
- ActionFormer: ~33M params (localization only)

1. At what metric levels does the "we do 5 tasks with one forward pass" story become compelling enough for publication?
2. Should we add GFLOPs/FPS benchmarks to the paper?

### F. Remaining Known Issues

| Issue | Status |
|-------|--------|
| Activity 1-class collapse (when enabled) | LDAM-DRW(s=30) suspected — Opus v4 says try plain CE + label smoothing first |
| PSR constant pattern | Fill-forward labels → near-constant targets → constant output near-optimal |
| Head pose NaN at val | Seen in Run 8 (AMP environment); check under FP32 |
| Mixup mixes logits, not inputs | Must rewrite before enabling |
| EMA shadow reset was no-op | Fixed (USE_EMA=False in recovery) |
| VideoMAE stream integration | Optional, adds ~80M params; skip for now? |
| Combined metric weighting | det=0.3, act=0.35, pose=0.15, psr=0.2 — is this right? |
| Val-line NaN for PSR/activity | Cosmetic: stub dict keys don't match Val line formatter expectations |

### G. Timeline to Paper-Ready

1. Rough estimate: how many GPU-days from here to numbers worth putting in a table?
2. Should we prioritize IndustReal first, then port to IKEA ASM?
3. Is the RTX 3060 (12GB) a bottleneck for full-dataset training?

---

## 4. Source Files (All Updated June 13, 16:20)

| File | Key Changes |
|------|-------------|
| `code/losses.py` | RC-28: empty frames skipped (lines 215-295) |
| `code/train.py` | RC-29: committed/skipped telemetry (lines 849-1560), PRE_VAL_GUARD isfinite fix |
| `code/config.py` | recovery_det_only + recovery presets, TRAIN_MAX_STEPS from env, SKIP_DET_METRICS_EVAL=False |
| `code/evaluate.py` | 5 guards: top5 shape check, TRAIN_ACT/PSR eval + logging skip |
| `code/model.py` | POPW architecture (~53M params), NaN guards |
| `code/optimizer.py` | AdamW + CosineAnnealingWarmRestarts |

## 5. Training Logs

| Log | Key Result |
|-----|-----------|
| `logs/recovery_r0_smoke.log` | R0 PASS: committed=55, skipped=0, 278s, det c 49→0.05 |
| `logs/recovery_r1_det_bootstrap.log` | R1 v4: det_mAP50=0.0091 (epoch 0), epoch 1 in progress |
| `logs/recovery_train8_run8.log` | Run 8 (pre-fix, for reference) |

## 6. Documentation

| File | Contents |
|------|----------|
| `13_OPUS_ANSWER_v4.md` | Your previous answer (RC-28/RC-29 diagnosis, R0-R3 protocol) |
| `14_POST_OPUS_V4_IMPLEMENTATION.md` | Full implementation journal — all fixes, crashes, retries, breakthrough |
| `15_GIT_DIFF_SUMMARY.txt` | Complete git diff of all source changes |
| `00_JOURNEY_AND_STATUS.md` | Full project timeline Phases 1-9 |
| `popw_paper_improved.tex` | Target paper with all SOTA benchmarks |

---

**The question**: We have a 53M-param unified model that just produced its first non-zero detection metric after fixing two critical bugs. What is the most efficient path to filling every `\popwres` in the paper with a number we can defend?
