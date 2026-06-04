# POPW Architecture Verification — Master Prompt for Claude Code

## GOAL

Verify that the POPW multi-task architecture implementation in this repository exactly matches the POPW paper (`popw_paper.tex`) and will produce results comparable to the IndustReal benchmark baselines when trained correctly.

---

## CONTEXT: WHAT POPW IS

POPW is a unified multi-task architecture for egocentric assembly understanding. It performs **five tasks in a single forward pass**:

1. **Assembly State Detection (ASD)** — 24 classes, RetinaNet-style head
2. **Body Pose** — 17 keypoints, IKEA ASM only (soft-argmax)
3. **Head Pose** — 9-DoF, IndustReal only
4. **Activity Recognition** — 74 classes, TCN + 2×ViT temporal blocks
5. **Procedure Step Recognition (PSR)** — 11 components, causal transformer

**Key architectural innovations:**
- ConvNeXt-Tiny backbone + FPN neck (shared across all tasks)
- Two-stage FiLM conditioning: PoseFiLM (body keypoints → C5) → HeadPoseFiLM (head pose → modulated C5)
- Kendall homoscedastic uncertainty weighting for multi-task loss balancing
- Staged training: Stage 1 (det-only), Stage 2 (+pose), Stage 3 (all tasks)

---

## FILES TO ANALYZE

### Primary Reference (read first)
| File | Lines | Path |
|------|-------|------|
| `popw_paper.tex` | 779 | `../popw_paper.tex` (up one level from impl/) |

### Implementation (verify against paper)
| File | Lines | Path |
|------|-------|------|
| `model.py` | 1725 | `./model.py` |
| `config.py` | 563 | `./config.py` |
| `losses.py` | 712 | `./losses.py` |
| `train.py` | 2000 | `./train.py` |
| `evaluate.py` | 2225 | `./evaluate.py` |
| `industreal_dataset.py` | ~900 | `./industreal_dataset.py` |

### Dataset
| File | Purpose |
|------|---------|
| `industreal_dataset.py` | ~900 | Dataset loader for IndustReal (IndustRealDataset class) |

---

## YOUR VERIFICATION TASKS

### PHASE 1: Architecture Verification

For each component below, read the paper section AND the corresponding implementation, then state whether they match exactly or have discrepancies.

**1.1 Backbone + FPN (Paper §Architecture / model.py lines 142–369)**

Verify:
- [ ] ConvNeXt-Tiny (ImageNet pretrained), channels C2=96, C3=192, C4=384, C5=768
- [ ] FPN takes [C3, C4, C5] → [P3, P4, P5, P6, P7], each 256 channels
- [ ] P6/P7 generated via stride-2 conv on C5
- [ ] Input resolution: 1280×720 (IndustReal), output stride 32

**1.2 Detection Head (Paper §Detection / model.py lines 418–530)**

Verify:
- [ ] RetinaNet-style, operates on P3–P7
- [ ] Shared cls/reg subnets: 4× Conv3×3 + ReLU → final conv
- [ ] Classification: 9 anchors × 24 classes, focal loss (α=0.25, γ=2)
- [ ] Regression: 9 anchors × 4 (bbox deltas), GIoU loss
- [ ] Anchor sizes: 3 ratios × 3 scales = 9 anchors per spatial location

**1.3 Pose Head (Paper §Body Pose / model.py lines 480–528)**

Verify:
- [ ] Input: P3 feature (stride 8, 256ch)
- [ ] ConvTranspose2d(k=4,s=2,p=1) + GroupNorm(32) + ReLU → upsampled resolution
- [ ] Conv1×1 → [B, 17, H/8, W/8] heatmaps
- [ ] Soft-argmax (T=0.1) → [B, 17, 2] keypoints + [B, 17] confidence
- [ ] Wing Loss (ω=0.05, ε=0.005), confidence-weighted

**1.4 Head Pose Head (Paper §Head Pose / model.py lines 1217–1253)**

Verify:
- [ ] GAP(C4) ‖ GAP(C5) → concat → MLP
- [ ] MLP: 1152→512→256→9 (with LayerNorm + GELU + Dropout)
- [ ] Output: 9-DoF = forward[3] ‖ position[3] ‖ up[3]
- [ ] Loss: MSE × 0.001 (meter-scale normalization)

**1.5 PoseFiLM (Paper §PoseFiLM / model.py lines 532–607)**

Verify:
- [ ] keypoints[B,34] ‖ confidence[B,17] → pose_flat[B,51]
- [ ] γ-net: 51→512→768, output 1+tanh ∈ (0,2)
- [ ] β-net: 51→512→768, output unbounded (linear)
- [ ] C5_direct = backbone C5 (bypasses FPN) [B,768,23,40]
- [ ] C5_mod = γ · C5_direct + β

**1.6 HeadPoseFiLM (Paper §HeadPoseFiLM / model.py lines 609–???)**

Verify:
- [ ] Input: head_pose[B,9] (stop_grad)
- [ ] γ_hp-net: 9→256→768, output 1+tanh ∈ (0,2)
- [ ] β_hp-net: 9→256→768, output unbounded
- [ ] C5_mod2 = γ_hp · C5_mod + β_hp
- [ ] GAP(C5_mod2) feeds into activity head

**1.7 Activity Head (Paper §Activity / model.py lines 1069–1215)**

Verify:
- [ ] det_conf = MaxPool(cls_preds) → [B,24] (stop_grad)
- [ ] f_joint = [det_conf(24) ‖ GAP(C5_mod2)(768) ‖ GAP(P4)(256)] → [B,1048]
- [ ] W_proj: Linear(1048→512) → f̃_t[B,512]
- [ ] Feature Bank: sliding window T=16, [B,16,512]
- [ ] TCN: 1D Depthwise Conv(k=5, dilation=1) → LayerNorm → GELU → DropPath=0.1
- [ ] 2× ViT blocks: CLS token + learnable pos embed + MHSA(8heads, d_k=64) + FFN(512→2048→512), DropPath 0.10/0.15, pre-norm
- [ ] CLS readout → Dropout(0.1) → Linear(512→74)
- [ ] Loss: LDAM-DRW (74 cls, label_smoothing=0.1)

**1.8 PSR Head (Paper §PSR / model.py lines 1258–1413)**

Verify:
- [ ] Multi-scale input: GAP(P3+P4+P5) → concat → MLP(768→256)
- [ ] Causal Transformer: 3 layers, 4 heads, d_model=256
- [ ] Per-component output heads: 11 separate tiny MLPs
- [ ] Loss: Binary Focal (α=0.25, γ=2.0) + temporal smoothness (w=0.05)

---

### PHASE 2: Loss & Training Verification

**2.1 Kendall Homoscedastic Uncertainty (Paper §Kendall / losses.py lines ???)**

Verify:
- [ ] Formula: L = Σ_t exp(-s_t) · L_t · ramp_t + s_t
- [ ] s_det, s_act, s_psr init=0; s_pose init=-1
- [ ] ramp_t: min(1, epoch/5) for activity; others ramp from epoch 1
- [ ] s_t clamped to [-4, 2]

**2.2 Staged Training Schedule (Paper §Training / train.py lines 372–455)**

Verify:
- [ ] Stage 1 (epochs 1–5): Detection only; backbone stages[0–1] frozen
- [ ] Stage 2 (epochs 6–15): + Body Pose + Head Pose; stages[0] frozen
- [ ] Stage 3 (epoch 16+): All tasks active; all layers trainable
- [ ] EMA: disabled in stage 1–2, enabled in stage 3 (decay=0.999)

**2.3 Loss Magnitudes (Paper §Loss scales)**

Verify scales are reasonable:
- [ ] Detection loss ~1–10 (focal+giou)
- [ ] Pose loss scaled ×0.001 (wing loss ~100→0.1)
- [ ] Head pose loss scaled ×0.001 (MSE meter-scale)
- [ ] Activity loss ~1–5 (LDAM-DRW)
- [ ] PSR loss ~0.1–1 (binary focal)

---

### PHASE 3: Benchmark Comparability

The paper reports these baselines on **IndustReal**:

| Task | Method | Score |
|------|--------|-------|
| ASD | YOLOv8m (COCO+synth+real) | **83.80% mAP@0.5** (bbox), **64.1% video-level mAP@0.5** |
| Activity | MViTv2 (K400, RGB+VL+stereo) | **65.25% Top-1** / **87.93% Top-5** |
| PSR | B2 ASD-accumulation | **POS=0.816**, **F1=0.731** |
| PSR | STORM-PSR dual-stream | **POS=0.812**, **F1=0.506**, τ=15.5s |

After training completes, POPW should be **competitive** on these numbers.

**3.1 Expected POPW performance ranges (your estimate based on architecture):**
- ASD mAP@0.5: ___ (target: comparable to YOLOv8m 83.80%)
- Activity Top-1: ___ (target: comparable to MViTv2 65.25%)
- PSR F1@±3f: ___ (target: comparable to B2 baseline 0.731)
- PSR POS: ___ (target: comparable to B2 baseline 0.816)

Note: POPW uses RGB-only (no VL+stereo) so activity may be slightly lower than MViTv2.

---

### PHASE 4: Efficiency Targets (Paper §Efficiency)

| Metric | Target (from paper) |
|--------|---------------------|
| Parameters | < 50M (estimate — ConvNeXt-Tiny ~28M + heads) |
| GFLOPs | ~200–300G for 1280×720 input |
| FPS (RTX 3060) | > 10 FPS streaming, > 30 FPS batched |
| Memory (batch=1) | < 10GB VRAM on RTX 3060 |

**4.1** Estimate total parameters by summing:
- ConvNeXt-Tiny backbone: ~28M
- FPN: ~3M
- Detection head: ~2M
- Pose head: ~0.5M
- PoseFiLM + HeadPoseFiLM: ~2M
- Activity head (TCN+ViT+proj): ~15M
- PSR head (transformer+per-component): ~1M
- **Total: ~52M** (give or take)

**4.2** Confirm batch size 2 fits in 12GB RTX 3060 with:
- Gradients + optimizer states + model weights + activations
- Mixed precision (FP16) training enabled

---

## QUESTIONS TO ANSWER

After reading all files, answer:

1. **ARCHITECTURE**: Does `model.py` exactly match the architecture described in `popw_paper.tex`? List any discrepancies (even minor ones).

2. **LOSSES**: Do the loss functions in `losses.py` exactly implement what the paper specifies? Note any differences in hyperparameters.

3. **TRAINING**: Does `train.py` implement the staged training schedule correctly? Does the Kendall weighting logic match the paper?

4. **BENCHMARK**: When POPW is trained to completion, what ASD mAP@0.5, Activity Top-1, and PSR F1 do you expect? Justify based on architecture differences from baselines.

5. **EFFICIENCY**: Will POPW meet the paper's efficiency targets (params, GFLOPs, FPS)? Estimate for batch=1 and batch=8 on RTX 3060.

6. **RISKS**: What are the top 3 risks that could prevent POPW from matching benchmark performance?

7. **FIXES NEEDED**: List any changes required to make the implementation exactly match the paper (if any).

---

## HOW TO RUN THIS ANALYSIS

1. Read `popw_paper.tex` first (the authoritative reference)
2. Then read each implementation file in order: `model.py`, `config.py`, `losses.py`, `train.py`
3. Compare each section against the paper
4. Run any quick sanity checks (e.g., count parameters with a Python snippet)
5. Compile your findings into answers for each question above

Be thorough. If something doesn't match, say so explicitly. This analysis must be rigorous because the whole point of POPW is to match the paper's competitive benchmark claims.
