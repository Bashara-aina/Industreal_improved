# AAIML 2027 -- Competitor Analysis

**Paper**: POPW: A Multi-Task Deep Learning Framework for Assembly Verification
**Comparison scope**: All systems mentioned in the paper plus additional baselines

---

## Summary Table

| System | Tasks | GPU Cost | Multi-Task | Ethics Framework | Factory Pilot | Code Public |
|--------|-------|----------|------------|------------------|---------------|-------------|
| **POPW (ours)** | **5 tasks**: detection, body pose, head pose, activity, PSR | **$299** | **Yes** (shared backbone, FiLM) | **IEEE 7005-2021** | **Yes (n=20)** | **Yes** (GitHub) |
| YOLOv8m | 1 task: detection | $299 | No | No | No | Yes |
| MViTv2-S | 1 task: action recognition | $2K+ GPU | No | No | No | Yes |
| STORM-PSR | 1 task: procedure step rec | $299 | No | No | No | Yes |
| ViMAT | 1 task: detection (matting) | $10K+ | No | No | No | No |
| IFAS | 1 task: screw fastening | $15K+ | No | No | No | No |
| Li et al. (2025) | 2 tasks: defect detection | $1K | Yes (2-task) | No | No | No |
| Traditional multi-model | 3-5 separate models | $12K-$55K | No (separate) | No | Varies | Mixed |

---

## Detailed Competitor Profiles

### YOLOv8m (Detection Only)

**Cost**: $299 (consumer GPU), open source

**Performance on IndustReal**: mAP50 0.838

**Advantages over POPW**:
- Much higher detection accuracy (2.5x on present-class mAP50, 3.8x on standard mAP50)
- Well-established, documented, deployment-tested
- 79 GFLOPs (lower than POPW's 93)

**Disadvantages vs POPW**:
- Single task only. Achieving all five tasks requires 3-5 separate models
- Uses 260K synthetic images + COCO pretraining (not directly comparable)
- No worker tracking, no pose, no activity, no procedure step recognition
- No ethics framework or worker compensation mechanism

**Key differentiator**: YOLOv8m wins on detection alone. POPW wins on task coverage. The right comparison is YOLOv8m * 5 models vs POPW * 1 model.

---

### MViTv2-S (Action Recognition Only)

**Cost**: Requires GPU with >8GB VRAM for training (approximately $2K+ workstation GPU)

**Performance on IndustReal**: Not directly benchmarked on the same dataset splits. STORM-PSR reports action recognition results.

**Advantages over POPW**:
- State-of-the-art video architecture
- Higher Top-1 accuracy for action recognition (when trained alone)

**Disadvantages vs POPW**:
- 170 GFLOPs (1.8x POPW's total for all 5 tasks)
- Video-only model -- no detection, no pose, no PSR
- Requires additional models for complete assembly understanding

**Key differentiator**: MViTv2-S represents the compute-expensive specialist approach that POPW replaces.

---

### STORM-PSR (Procedure Step Recognition)

**Cost**: $299 (consumer GPU)

**Performance on IndustReal**: Established PSR benchmark results

**Advantages over POPW**:
- Specialized architecture for procedure step recognition
- Higher PSR accuracy (dedicated, not multi-task)

**Disadvantages vs POPW**:
- 112 GFLOPs (higher than POPW's 93 GFLOPs for ALL tasks)
- 28.4M params (only for PSR; POPW does 5 tasks for 53M)
- Single-task only
- No detection, no pose, no activity (separate from PSR)

**Key differentiator**: POPW's PSR head is a lightweight causal Transformer (3 layers, 4 heads, d=256) designed to be efficient as part of a multi-task system, not to beat a specialized PSR model.

---

### ViMAT (Visual Matting for Assembly)

**Venue**: ICIAP 2025

**Description**: Visual matting-based approach for assembly component detection. Single-task system.

**Cost**: $10K+ GPU hardware

**Comparison**: ViMAT is more expensive and handles only detection. POPW handles 5 tasks at 3% of the hardware cost.

---

### IFAS (Intelligent Fastening Assembly System)

**Venue**: Journal of Intelligent Manufacturing, 2026

**Description**: Specialized system for screw fastening verification. Uses depth sensors and a dedicated CNN.

**Cost**: $15K+ (industrial sensors + GPU)

**Comparison**: IFAS targets a single assembly operation (screw fastening) with specialized hardware. POPW targets general assembly verification with a single RGB camera. IFAS is narrower but potentially more accurate for its specific domain.

---

### Li et al. (2025) -- Multi-Task Defect Detection

**Venue**: Machines, 2025

**Description**: Two-task Swin Transformer for defect detection. Closest prior work on multi-task for manufacturing.

**Cost**: Approximately $1K GPU

**Limitations**:
- Two tasks only (vs five in POPW)
- Defect detection only, not assembly verification
- No temporal modeling (frame-by-frame only)
- No blockchain or ethics framework
- No factory pilot

**Key differentiator**: Li et al. show multi-task is feasible for manufacturing. POPW extends from 2 to 5 tasks across detection, pose, activity, and procedure recognition.

---

### Traditional Multi-Model Approach

**Configuration**: 3-5 separate models running on separate GPUs or a time-shared workstation

**Typical setup**:
- YOLOv8m for detection ($299 GPU or cloud)
- MViTv2-S for action recognition ($2K+ GPU)
- HRNet for pose estimation ($299 GPU)
- STORM-PSR for procedure steps ($299 GPU)
- Head pose model ($299 GPU or part of activity model)

**Total cost**: $12,000-$55,000 depending on GPU count

**Total compute**: 285+ GFLOPs per frame (79+170+36+112...)

**Key differentiator**: POPW reduces hardware cost by 97% and compute by 68% while adding a unified verification pipeline.

---

## Where POPW Wins

1. **Task coverage**: No other single system covers 5 assembly understanding tasks
2. **Hardware cost**: $299 vs $12K-$55K for equivalent multi-model setup
3. **Verification pipeline**: End-to-end from camera to blockchain payment
4. **Ethical governance**: Only system with IEEE 7005 framework and pilot validation
5. **Worker compensation**: Only system with automatic micropayments per task

## Where POPW Loses

1. **Per-task accuracy**: Each individual task head underperforms its specialized counterpart
2. **Detection mAP**: 0.34 vs 0.838 (YOLOv8m) -- 59% relative gap
3. **Activity recognition**: 18.3% vs MViTv2-S (estimated 30-40% on same data)
4. **Speed**: 4.8 FPS vs specialist models that can run at 15-30 FPS on appropriate hardware
5. **Maturity**: New architecture vs established models with production deployments

---

## Competitive Positioning for the Paper

The paper should position POPW as:

> **"A unified, affordable alternative to the multi-model approach, not a replacement for specialist models."**

The core argument is: if you need all five tasks (common in assembly verification), POPW is:
- 97% cheaper ($299 vs $12K-$55K)
- 68% less compute (93 vs 285+ GFLOPs)
- Single-coordination overhead (no multi-model fusion)
- Ethics-governed and pilot-validated

The tradeoff is per-task accuracy, which is acceptable because:
- Detection errors are mostly 1-bit-adjacent (coarse state is correct)
- Activity is per-frame, and PSR uses temporal smoothing
- The system as a whole is sufficient for the downstream task

---

## Data for Cost Comparison Figure

| System Category | Min Cost | Max Cost | Midpoint | Tasks | GFLOPs |
|----------------|----------|----------|----------|-------|--------|
| Traditional multi-model | $12,000 | $55,000 | $33,500 | 3-5 | 361+ |
| ViMAT | $10,000 | $15,000 | $12,500 | 1 | ~100 |
| IFAS | $15,000 | $20,000 | $17,500 | 1 | ~50 |
| Li et al. (2-task) | $1,000 | $1,500 | $1,250 | 2 | ~80 |
| YOLOv8m (detection only) | $299 | $299 | $299 | 1 | 79 |
| **POPW (5-task)** | **$299** | **$799** | **$549** | **5** | **93** |
