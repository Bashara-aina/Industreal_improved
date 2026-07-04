# Reviewer 5: Ablation Matrix & Multi-Task Efficiency Proof

## Identity: IEEE/CVF Reviewer — Multi-Task Learning & Efficient Architectures
**Focus:** Ablation rigor, multi-task interference quantification, parameter efficiency claims.
**Bias:** Will not accept "efficiency" claims without Ablation A (single-task vs multi-task on same backbone). Demands matched-parameter, matched-data comparisons.

---

## 1. The Ablation Matrix

This is what the paper MUST have to survive peer review. Every cell must be filled.

### Current Status

| Ablation | Status | Time Needed |
|---|---|---|
| **A: Single-task vs multi-task (SAME backbone, SAME data)** | 🔴 Not run (ablation_det_only running on 3060) | ~12h remaining for det |
| **B: Kendall vs fixed weights** | 🔴 Not run | ~2 days |
| **C: Verb-grouping vs raw 75 classes** | 🔴 Not run | ~2 days |
| **D: EMA on/off** | 🟡 Already have EMA=0.995 | 1 comparison (reload checkpoint) |
| **E: FiLM on/off** | 🔴 Not run | ~2 days |
| **F: PSR seq_every 4 vs 2** | 🟡 F7 fix changes this | Historical comparison only |

---

## 2. Ablation A — The Paper's Scientific Core

This is the single most important experiment. Without it, the paper has no thesis — just a collection of metrics.

### What It Proves

| Question | Answered By |
|---|---|
| Does multi-task training hurt individual heads? | Single-task mAP vs multi-task mAP |
| How much efficiency do we actually gain? | Parameter count: (sum of single-task params) vs (multi-task total params) |
| Is the efficiency worth the accuracy cost? | The central paper thesis |

### Expectation

| Head | Single-Task (estimated) | Multi-Task (epoch 11) | Multi-Task Cost |
|---|---|---|---|
| Detection mAP@0.5 | ~0.45 | **0.317** | **-29%** |
| Activity macro-F1 | ~0.15 | **0.110** | **-27%** |
| PSR comp acc | ~0.50 | **0.346** | **-31%** |
| Pose fwd MAE | ~7° | **8.14°** | **+16%** |

**Expected story:** Multi-task costs 15-30% per-head accuracy but delivers 4 tasks for 1 model at 1/3 the total parameters.

### The Efficiency Table

| Metric | 4× Single-Task | POPW Multi-Task | Savings |
|---|---|---|---|
| Total parameters | ~112M (4 × 28M) | **~28M** | **75% fewer** |
| Inference passes | 4 | **1** | **75% fewer** |
| GPU memory (training) | ~32 GB | **~9 GB** | **72% less** |
| GPU cost (hardware) | $1,716 (4×$429) | **$429** | **75% cheaper** |
| Det mAP@0.5 | ~0.45 | **0.317** | -29% |
| Act macro-F1 | ~0.15 | **0.110** | -27% |

**The paper writes itself from this table.**

---

## 3. Ablation B: Kendall vs Fixed Weights

**Protocol:** Run `KENDALL_FIXED_WEIGHTS=1` with stage_rf4 preset. Compare validation metrics at matched epochs.

**The ablation preset exists** (`run_ablation_suite.sh kendall-fixed`). It's a 25-epoch run.

**Expected outcome:** Kendall (learned) should outperform fixed weights by ~0.02-0.05 combined metric. If it doesn't, the Kendall framework isn't adding value and the paper should note this honestly.

---

## 4. Ablation C: Verb-Grouping vs Raw 75 Classes

**Protocol:** Run `ACT_CLASS_GROUPING=none` with stage_rf4 preset.

**Why this matters:** If raw 75 classes achieves similar macro-F1, the verb-grouping is unnecessary complexity. If raw 75 collapses (more classes, harder), the grouping is validated.

**Expected:** Raw 75 will have lower macro-F1 (50% more classes to discriminate) but potentially higher top-5 accuracy (more fine-grained distinctions). The trade-off defines whether our verb-grouping is a contribution or a workaround.

---

## 5. Parameter Efficiency Claim — The Arithmetic

Our current claim: **"31% fewer parameters than the 3-model pipeline."**

**This is wrong and must be fixed before submission.**

| Model | Params | Notes |
|---|---|---|
| YOLOv8-m (detection) | 25M | SOTA for our task |
| MViTv2-S (activity) | 36M | SOTA for activity |
| B3/STORM-PSR | ~25M (YOLOv8m again) | Reuses detection backbone |
| **Total pipeline** | **~86M** | But YOLOv8m counted twice |
| **Our model** | **~28M** | ConvNeXt-Tiny |
| **Our savings vs 3-model pipeline** | **~67%** | Not 31% |

**Corrected claim:** *"Our single model requires approximately 67% fewer parameters than the dedicated multi-model pipeline, while processing all 4 tasks in a single forward pass."*

---

## 6. Efficiency Realities

### Inference Speed

We claim "real-time" but have NEVER MEASURED FPS. The ablation config `SKIP_EFFICIENCY_METRICS=True` means our eval skips FPS/FLOPs measurement.

**Fix:** Add FPS measurement on the 5060 Ti:
- Batch = 1 (realistic deployment)
- Measure: time per forward pass across 1000 frames
- Report: FPS, total FLOPs (via fvcore or ptflops)

### GPU Cost Claim

We claim "$299 GPU" but the RTX 5060 Ti costs **$429 MSRP**. The $299 price is for the RTX 3060.

**Fix:**
- If running on 5060 Ti: claim "$429" or "consumer GPU ($429)"
- If comparing to V100 ($2,500+): still a 6x cost reduction

---

## 7. Required Ablation Plan (Priority Order)

| P0 | Ablation A: Single-task runs | Already started | Paper won't survive review without it |
|---|---|---|---|
| P0 | Fix FPS/FLOPs measurement | 1h | Efficiency claim needs numbers |
| P0 | Fix "31% fewer params" arithmetic | Today | Current claim is wrong |
| P1 | Ablation B: Kendall vs fixed | 2 days | Validates multi-task balancing |
| P1 | Ablation C: Verb-grouping vs raw | 2 days | Validates grouping protocol |
| P2 | Ablation E: FiLM on/off | 2 days | Validates architectural contribution |
| P2 | FPS on 3060 vs 5060 Ti | 1h | Complete efficiency picture |

---

## 8. Reviewer Bottom Line

**From a pure ML systems perspective, this paper has a defensible thesis if, and only if:**

1. ✅ Ablation A is complete and shows the multi-task cost explicitly
2. ✅ Efficiency numbers (params, FPS, GPU cost) are measured, not estimated
3. ✅ Each head's comparison is honest about what task it's actually solving
4. ✅ The parameter arithmetic is corrected (31% → ~67%)

Without (1), the paper is a demo, not a scientific contribution. **Ablation A is the single highest-priority item in the entire project.**
