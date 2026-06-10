# POPW Architecture Review — Master Prompt for Claude

**For**: Claude Code analysis of POPW multi-task architecture
**Date**: 2026-05-07
**Goal**: Verify POPW implementation matches `popw_paper.tex` claims — architecture correctness, efficiency figures, and readiness for training

---

## Your Task

You are reviewing the POPW (Pose-Only Pose-w Mehr) multi-task egocentric video understanding architecture. Your job is to:

1. Read `popw_paper.tex` and extract all architecture claims, benchmark numbers, and efficiency specs
2. Read the implementation files listed below
3. Verify each claim against actual source code
4. Flag any discrepancy (architectural, parametric, or behavioral) between the paper and the implementation
5. State whether the implementation is ready for full training

---

## Files to Analyze

### Primary Source of Truth: Paper
```
/home/newadmin/swarm-bot/project/popw/working/code/popw_paper.tex
```

### Implementation Files (in order of importance)
```
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/model.py      # POPWMultiTaskModel — the full architecture
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/evaluate.py  # Metrics — all 5 task metrics
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/losses.py     # All loss functions
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/config.py     # Hyperparameters
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/train.py       # Full multi-task training loop
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/smoke_test.py  # 14-test verification suite (already passing)
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/pretrain_synthetic.py  # Detection pretrain (already verified 1-epoch run)
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/industreal_dataset.py  # Dataset loader
```

### Verification Reports (context — do not use as primary source)
```
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/POPW_FINAL_PRETRAIN_VERIFICATION.md
```

---

## Paper Claims to Verify

### A. Architecture Specifications (from popw_paper.tex §Proposed Approach)

#### A1. Backbone — ConvNeXt-Tiny + FPN
- **Paper claim**: ConvNeXt-Tiny pretrained on ImageNet. Input [B, 3, 720, 1280] produces C2(stride4, 96ch), C3(stride8, 192ch), C4(stride16, 384ch), C5(stride32, 768ch).
- **FPN**: Lateral 1×1 conv (192/384/768 → 256), top-down upsampling, 3×3 smoothing, P6/P7 from stride-2 conv on C5. Outputs {P3,P4,P5,P6,P7}, each 256ch.
- **Verify**: Check `model.py` — ConvNeXt-Tiny instantiation, stage_to_features, FPN construction, channel dimensions at each level.

#### A2. Detection Head (24 ASD classes)
- **Paper claim**: RetinaNet-style head on P3–P7. Cls subnet: 4× Conv3×3+ReLU → Conv(9×24) → cls_preds [B,N,24]. Reg subnet: 4× Conv3×3+ReLU → Conv(9×4) → reg_preds [B,N,4]. Anchors: 3 ratios × 3 scales, sizes (24,48,96,192,384), k-means calibrated.
- **Loss**: Focal loss (α=0.25, γ=2) + GIoU loss.
- **Verify**: Check `model.py` — detection head architecture, anchor generation, loss computation in `losses.py`.

#### A3. Body Pose Head (17 keypoints — IKEA ASM)
- **Paper claim**: ConvTranspose2d (k=4, s=2, p=1) + GroupNorm(32) + ReLU → [B,256,180,320]. Conv1×1 → [B,17,180,320] (heatmaps). Soft-argmax (T=0.1) → keypoints [B,17,2] + conf [B,17].
- **Loss**: Wing Loss (ω=0.05, ε=0.005), confidence-weighted.
- **Verify**: Check `model.py` — upsample layers, heatmap head, soft-argmax implementation, `losses.py` Wing Loss parameters.

#### A4. Head Pose Head (9-DoF — IndustReal only)
- **Paper claim**: GAP(C4) ‖ GAP(C5) → [B, 384+768=1152]. MLP: 1152→512→256→9 (LayerNorm+GELU+Dropout). Output: head_pose [B,9] = forward[3] ‖ position[3] ‖ up[3]. Loss: MSE × 0.001 (meter-scale normalization).
- **Verify**: Check `model.py` — HeadPoseMLP architecture, input concatenation, output dimension, loss scaling in `losses.py`.

#### A5. Activity Recognition Head (74 classes — paper says 74, implementation uses 75)
- **Paper claim**: Detection context: MaxPool(cls_preds) → f_det [B,24], stop_grad. Spatial: GAP(C5_mod2) [B,768] (after FiLM) ‖ GAP(P4) [B,256]. Joint: Concat [f_det, f_app, f_spatial] → f_joint [B,1048]. Projection: W_proj (1048→512) → f̃_t [B,512]. Feature Bank: sliding window B_t = [f̃_{t-T+1},...,f̃_t] [B,T=16,512], keyed by (video_id, camera_view), 16 KB/seq (FP16).
- **TCN Block**: 1D Depthwise Conv (k=5, dilation=1). LayerNorm → GELU → Linear. DropPath=0.1.
- **ViT Temporal Blocks** (2 layers): Prepend CLS token [1,1,512]; Learnable pos embed [1,T+1,512]; MHSA (8 heads, d_k=64, attn_dropout=0.1); FFN (LayerNorm → Linear 512→2048 → GELU → Linear 2048→512); DropPath 0.10, 0.15; pre-norm.
- **Output**: cls_token → act_logits [B,74] (or [B,75] for IndustReal — see note below).
- **Loss**: LDAM-DRW Loss (74 cls, label_smooth=0.1).
- **Note**: The paper says 74 activity classes for IndustReal (line 278: "74 atomic action classes"), but the implementation has NUM_CLASSES_ACT=75 (IDs 0-74). Verify which is correct by checking the actual dataset label file. The IndustReal AR_labels.csv may have 75 classes (IDs 0-74). If so, this is a dataset documentation issue, not an implementation bug.
- **Verify**: Check `model.py` — activity head architecture (C5_mod2, P4 GAP, detection context, projection, feature bank, TCN, ViT). Check `losses.py` — LDAMLoss label_smoothing=0.1.

#### A6. PSR Head (11 components)
- **Paper claim**: Binary Focal Loss (α=0.25, γ=2.0) + temporal smoothness (w=0.05).
- **Verify**: Check `model.py` — PSR head architecture, `losses.py` — BinaryFocalLoss parameters, temporal smoothness weight.

#### A7. PoseFiLM (1st stage — body keypoints)
- **Paper claim**: Confidence: heatmaps → max → sigmoid → nan_to_num(0.5); no gradient. Pose encoding: keypoints [B,34] ‖ confidence [B,17] → pose_flat [B,51]. γ-net: 51→512→768, output 1+tanh(·) ∈ (0,2). β-net: 51→512→768, output unbounded. C5_direct: from backbone (bypasses FPN) [B,768,23,40]. Modulation: C5_mod = γ·C5_direct + β [B,768,23,40].
- **Verify**: Check `model.py` — PoseFiLMγnet, PoseFiLMβnet architectures, input construction, modulation operation, stop_grad on confidence.

#### A8. HeadPoseFiLM (2nd stage — 9-DoF)
- **Paper claim**: Input: head_pose [B,9] (stop_grad). γ_hp-net: 9→256→768, output 1+tanh(·). β_hp-net: 9→256→768, output unbounded. Modulation: C5_mod2 = γ_hp·C5_mod + β_hp [B,768,23,40]. GAP(C5_mod2) feeds activity head.
- **Verify**: Check `model.py` — HeadPoseFiLMγnet, HeadPoseFiLMβnet architectures, second modulation, stop_grad on head_pose.

### B. Loss Functions (from popw_paper.tex §Multi-Task Loss)

| Loss | Formula | Parameters | Verified? |
|------|---------|------------|-----------|
| Detection | Focal(α=0.25, γ=2) + GIoU | — | |
| Body Pose | Wing Loss (ω=0.05, ε=0.005) × 0.001 | meter scale | |
| Head Pose | MSE × 0.001 | meter scale | |
| Activity | LDAM-DRW, label_smooth=0.1 | 74/75 classes | |
| PSR | BinaryFocal(α=0.25, γ=2.0) + temporal_smooth(w=0.05) | 11 components | |

**Kendall uncertainty weighting**: L = Σ_t exp(-s_t) · L_t · ramp_t + s_t, where s_t = clamp(log σ²_t, -4, 2).
**Init**: s_det=0, s_pose=-1, s_act=0, s_psr=0.

**Staged training**:
- Stage 1 (epochs 1–5): Detection only; backbone layer1–3 frozen.
- Stage 2 (epochs 6–15): + Pose + HeadPose; Activity and PSR frozen.
- Stage 3 (epoch 16+): All four task groups active.

**Activity ramp**: min(1, epoch/5).

### C. IndustReal Benchmarks to Beat (from popw_paper.tex Table 7)

| Task | Baseline | Metric | Target |
|------|----------|--------|--------|
| Assembly State Detection | YOLOv8m (COCO+synth+real) | mAP (b-boxed) | 83.80% |
| Assembly State Detection | YOLOv8m (COCO+synth+real) | mAP@0.5 (all frames) | — |
| Activity Recognition | MViTv2 (Kinetics-400, RGB-only) | Top-1 | 65.25% |
| Activity Recognition | MViTv2 (K-400, RGB-only) | Top-5 | 87.93% |
| PSR | B2 ASD-accumulation baseline | F1 (±3-frame) | 0.731 |
| PSR | B2 ASD-accumulation baseline | POS | 0.816 |
| PSR | STORM-PSR | F1 (±3-frame) | 0.506 |
| Assembly State Recognition | SupCon+ISIL (ResNet-34) | F1@1 | ~0.85 |

**Important modality note** (paper line 457): The MViTv2 baseline uses RGB+VL+stereo depth as input. POPW uses RGB only. This is not a fair comparison for Top-1 — the VL and stereo modalities provide extra spatial/depth cues. Compare fairly or note the gap.

### D. Efficiency Targets (from popw_paper.tex Table 8)

Measured on **RTX 3060 (12GB)**:
- **Batched inference**: batch size 8
- **Streaming inference**: batch size 1

The paper has placeholder values (\popwres) for POPW's efficiency figures. The implementation's `compute_efficiency_metrics()` produces these measured numbers (RTX 3060, batch_size=1, 1280×720):

| Metric | Value (Measured) |
|--------|-----------------|
| Total params | **53.25M** |
| GFLOPs | **232.9G** |
| FPS (batched, bs=1) | **11.8 fps** |
| FPS (streaming) | **11.8 fps** |
| Pipeline params | 64.0M |
| Pipeline GFLOPs | 238.0G |
| Pipeline FPS | 15.0 fps |

Note: The pipeline figures (64M params, 238 GFLOPs, 15 fps) are estimates from a second model pass with doubled feature extraction. Verify these estimates against the paper's claims.

Compare these against the baselines in the paper's efficiency table (Table 8). The key comparison is POPW (single unified model, all 5 tasks) vs the dedicated baseline trio (YOLOv8m + MViTv2 + STORM-PSR running separately).

### E. Dataset Specs (from popw_paper.tex §Datasets)

**IndustReal**:
- Resolution: 1280×720
- Cameras: 1 RGB (egocentric)
- Detection classes: 24 ASD states
- Pose: head, 9-DoF
- Action classes: 74 (paper) vs 75 (implementation — verify)
- PSR: 11 components

---

## Verification Checklist

For each item below, state: **MATCH** / **DISCREPANCY** / **CANNOT VERIFY** (with reason)

### Architecture
- [ ] ConvNeXt-Tiny backbone, 4 stages, correct channel dims per stage
- [ ] FPN: lateral 1×1 convs, correct channel reduction, P6/P7 generation
- [ ] Detection head: RetinaNet-style, correct anchor config, 24 output classes
- [ ] Pose head: 17 keypoints, soft-argmax, Wing Loss params
- [ ] Head pose head: 9-DoF, MLP architecture, meter-scale loss
- [ ] Activity head: correct feature concat dims, TCN+ViT temporal, CLS token, 74/75 classes
- [ ] PSR head: 11 components, BinaryFocalLoss + temporal smoothness
- [ ] PoseFiLM: γ-net (51→512→768, tanh output), β-net, modulation, stop_grad
- [ ] HeadPoseFiLM: γ_hp-net (9→256→768, tanh), β_hp-net, second modulation, stop_grad
- [ ] Feature Bank: T=16, keyed by (video_id, camera_view), correct shape
- [ ] DropPath applied in TemporalConvBlock and ViTTemporalBlock

### Loss & Training
- [ ] Kendall uncertainty init: s_det=0, s_pose=-1, s_act=0, s_psr=0
- [ ] Focal Loss: α=0.25, γ=2
- [ ] Wing Loss: ω=0.05, ε=0.005
- [ ] LDAMLoss: label_smoothing=0.1
- [ ] BinaryFocalLoss: α=0.25, γ=2.0
- [ ] PSR temporal_smooth_weight=0.05
- [ ] Staged training: 3 stages with correct task group activation
- [ ] Activity ramp: min(1, epoch/5)
- [ ] EMA: decay=0.999, shadow model updated correctly
- [ ] BATCH_SIZE=2, GRAD_ACCUM_STEPS=16 (effective batch=32)

### Metrics
- [ ] Detection: mAP@0.5, mAP@[0.5:0.95], per-class AP
- [ ] Activity: Top-1, Top-5, frame accuracy, clip accuracy, macro-F1
- [ ] PSR: F1@±3, F1@±5, POS (at ±3), precision, recall
- [ ] Assembly State: F1@1, MAP@R(+)
- [ ] Head Pose: forward angular MAE (deg), up angular MAE (deg), position MAE (mm)
- [ ] Efficiency: params (M), GFLOPs, FPS (batched + streaming)

### Code Quality
- [ ] No hardcoded values — all via config.py
- [ ] Mixed precision (FP16) training enabled
- [ ] Gradient clipping configured
- [ ] Checkpoint saving/loading works
- [ ] pretrain_synthetic.py runs without error (already verified: 1-epoch run passed)

---

## Output Format

Provide your analysis in this structure:

### 1. Architecture Verification Summary
(MATCH / DISCREPANCY for each of the 20+ architecture items above)

### 2. Discrepancies Found
(For each discrepancy: what the paper says vs what the code does, severity: BLOCKING / WARNING / INFO)

### 3. Efficiency Analysis
- Compute the actual efficiency numbers from `compute_efficiency_metrics()` output
- Compare against the baseline efficiency table in the paper
- Note: the paper's efficiency table has \popwres placeholders — your job is to fill in whether the implementation can produce these numbers

### 4. Readiness Assessment
**READY** / **NOT READY** / **CONDITIONALLY READY**

If not ready, list the exact blocking issues that must be fixed before training.

### 5. Benchmark Expectation
Based on the architecture and the baseline numbers, what realistic accuracy would you expect POPW to achieve on IndustReal after full training? Give a per-task estimate with reasoning.

---

## Important Notes

- The paper has some placeholder values (\popwres) that need to be filled with actual measurements from the implementation. Verify the implementation CAN produce these numbers, even if they haven't been filled in the paper yet.
- The activity head uses 75 classes in the implementation (IDs 0-74) but the paper says 74. Check `industreal_dataset.py` and the actual AR_labels.csv to determine the ground truth.
- The paper's MViTv2 baseline for activity uses RGB+VL+stereo (multi-modal). POPW uses RGB only. Flag this modality gap when comparing Top-1 numbers.
- The paper's PSR POS definition (runs-based adjacent pair ordering) may differ from STORM-PSR's published code (Damerau-Levenshtein on action ID sequences). Verify the implementation's POS computation matches the paper's definition.
- There is a known config mismatch: `USE_VIDEOMAE=True` is defined in config.py but the model is built with `use_videomae=False`. This is intentional — VideoMAE is not used in the reported experiments.
- There is a known config mismatch: `USE_PSR_SEQUENCE_MODE=False` is defined but FeatureBank is used instead. Sequence mode is deferred.
- The head pose head is specific to IndustReal (9-DoF) and should not be activated for IKEA ASM.

---

## How to Run the Verification

```bash
# Smoke tests (already passing 14/14)
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive
python smoke_test.py

# Efficiency metrics
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive
python -c "
import torch, sys
sys.path.insert(0, '.')
from model import POPWMultiTaskModel
from evaluate import compute_efficiency_metrics
model = POPWMultiTaskModel(pretrained=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = compute_efficiency_metrics(model, input_size=(720, 1280), device=device)
for k, v in metrics.items():
    print(f'{k}: {v}')
"

# Pretrain (already verified — 1 epoch, 2 recordings, passed)
python pretrain_synthetic.py --epochs 1 --max-recordings 2
```
