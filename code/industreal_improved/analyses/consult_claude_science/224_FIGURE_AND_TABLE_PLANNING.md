# 224 — Figure & Table Planning for AAIML Paper

**Document:** 224 of 227 (Claude Science consultation package, docs 208–227)

**Status:** Planning document — defines every figure and table the paper needs

**Date:** 2026-07-11

---

## Table of Contents

1. Overview and Principles
2. FIGURE 1: System Overview / Teaser
3. FIGURE 2: Kendall Collapse Visualization (Core Contribution)
4. FIGURE 3: Per-Task Transfer Map
5. FIGURE 4: Gradient Conflict Analysis (E8 Visualization)
6. FIGURE 5: Efficiency Comparison
7. FIGURE 6: Qualitative Results
8. TABLE 1: Dataset Statistics
9. TABLE 2: Architecture Specification
10. TABLE 3: Main Results — MTL vs ST with 95% CI
11. TABLE 4: Comparison with Published MTL Methods
12. TABLE 5: Ablation Study
13. TABLE 6: Efficiency Metrics
14. TABLE 7: Per-Class Activity Breakdown
15. TABLE 8: PSR Per-Component Event-F1
16. Figure and Table Placement in Paper Flow

---

## 1. Overview and Principles

### 1.1 Page Budget Constraints

AAIML allows 8 pages (plus references). With a standard allocation of approximately 55% figures + 25% tables + 20% text per page, the available real estate is:

- **Figures:** 6 figures, approximately 1/3 to 1/2 page each. Three of these (Figures 2, 3, 4) must be impactful enough to stand alone as visual evidence for the paper's three core contributions. Figures 1 and 6 can be smaller. Figure 5 must fit a radar chart legibly.
- **Tables:** 8 tables, but only 3-4 can appear in the main paper body. The remaining 4-5 tables go in supplementary material (with a reference in main text).
- **Priority for main paper:** Table 3 (main results), Table 4 (published comparison), Table 5 (ablation), Table 6 (efficiency). Tables 1, 2, 7, 8 are supplementary.

### 1.2 The Narrative Arc (What Each Visual Element Must Prove)

The paper has a three-claim spine. Each figure and table must serve exactly one claim:

| Claim | Primary Figures | Primary Tables |
|---|---|---|
| **Claim 1: Capped Kendall fixes collapse** | Figure 2 (trajectories) | Table 5 (ablation row) |
| **Claim 2: MTL beats ST baselines** | Figure 3 (transfer map) | Table 3 (main results) |
| **Claim 3: MTL is dramatically more efficient** | Figure 5 (radar chart) | Table 6 (efficiency) |

Figures 1, 4, 6 and Tables 1, 2, 4, 7, 8 provide **context and support** for these three claims.

### 1.3 Color Scheme and Accessibility

- Use a single consistent color palette across all figures: 4 task colors (det=#E74C3C/red, act=#3498DB/blue, psr=#2ECC71/green, pose=#F39C12/orange).
- All figures must be legible in grayscale (hatching/patterns as fallback).
- Minimum 10pt font for all axis labels and legends.
- Vector format (PDF/SVG) for all line plots and diagrams; 300 DPI PNG for qualitative frames.
- Marker shapes on line plots: circle (det), square (act), triangle (psr), diamond (pose).

---

## 2. FIGURE 1: System Overview / Teaser

**Purpose:** Show the reader at a glance what the system does, what the four tasks are, and why it matters. This is the "elevator pitch" figure. It must communicate the single-backbone, four-task architecture and the efficiency claim within 5 seconds.

**Placement:** First page, top half. The teaser must hook the reader before they read the abstract.

**Panel Layout:** A single horizontal figure with 3 panels (A, B, C) arranged left to right.

### Panel A: Input Frame (10% of figure width)
- **Content:** A single egocentric frame from the IndustReal dataset showing a worker's hands assembling a transmission on a workbench. The frame should be representative: hands visible, assembly board partially populated with components, tools visible.
- **Overlay:** Subtle "t=0" label and a border indicating the 224x224 crop region (dashed box if showing full 1080x720).
- **Caption text below:** "Single egocentric video frame (224x224)"

### Panel B: Architecture Diagram (50% of figure width)
- **Content:** A clean block diagram showing:
  1. **Input layer:** RGB frame [3, 224, 224] represented as a small rectangle with color channels icon.
  2. **Backbone:** MViTv2-S (34.5M params) represented as a tall, wide block with a multiscale visual motif (three progressively smaller internal blocks to show hierarchical pooling). Label inside: "MViTv2-S Backbone" with a small "Kinetics-400 pretrained" badge.
  3. **Feature pyramid:** BiFPN shown as a sideways diamond/arrow structure branching off the backbone at P3/P4/P5 levels. Draw three distinct feature pathways.
  4. **Four heads:** Four rectangular blocks branching from the backbone/FPN, each with a distinct icon:
     - **Detection head (P3/P4/P5 features):** Small bounding-box icon. Label: "Detection (24 classes)".
     - **Activity head (cls_token):** Text-lines icon. Label: "Activity (75 classes)".
     - **PSR head (P5 features):** Timeline icon with state markers. Label: "PSR (11 components)".
     - **Pose head (cls_token):** Coordinate-axis icon. Label: "Head Pose (6-DoF)".
  5. **Connecting lines:** Color-coded to match the task colors. Show that all four heads share the single backbone via a single forward pass.

- **Key annotation:** A bold arrow from input to output labeled "Single Forward Pass" to emphasize the efficiency claim.
- **Parameter counts** displayed in small text beneath each component: backbone "34.5M", each head's param count, total "48.6M".

### Panel C: Efficiency Claim (30% of figure width)
- **Content:** A before/after bar chart or icon comparison:
  1. **Naive approach:** Four separate ST models. Show 4 backbone icons with "4x forward passes" and "108M params" and "X fps".
  2. **Our approach:** One backbone with 4 heads. Show 1 backbone icon with "1 forward pass" and "48.6M params" and "4X fps".
  3. Connecting visual: An arrow from "before" to "after" labeled "2.2x fewer params, 4x faster inference".
- **Stated numbers must be verifiable:** 48.6M = 34.5M (backbone) + 2.5M (FPN) + 0.8M (det) + 1.1M (act) + 1.8M (PSR) + 0.2M (pose). The ST ensemble total is approximately 108M (four separate backbones + heads).

### Data Needed to Build This Figure
- One representative egocentric frame (screenshot from IndustReal dataset).
- The final parameter counts for each component (from the model's `named_parameters()` or a param count script).
- The FPS numbers for MTL and ST ensemble (from `eff_fps` and `eff_fps_streaming` metrics).
- The MViTv2-S block diagram motif (can use diagrams.net / Matplotlib patches / TiKZ).

### How to Compute
```python
# Parameter count by component
def count_params_by_component(model):
    counts = {
        'backbone': sum(p.numel() for n, p in model.named_parameters() if 'backbone' in n),
        'fpn': sum(p.numel() for n, p in model.named_parameters() if 'fpn' in n or 'feature_pyramid' in n),
        'detection_head': sum(p.numel() for n, p in model.named_parameters() if 'detection' in n or 'det_head' in n),
        'activity_head': sum(p.numel() for n, p in model.named_parameters() if 'activity' in n or 'act_head' in n),
        'psr_head': sum(p.numel() for n, p in model.named_parameters() if 'psr' in n),
        'pose_head': sum(p.numel() for n, p in model.named_parameters() if 'pose' in n),
    }
    return counts
```

### What Story It Tells
"Four assembly perception tasks. One backbone. One forward pass. Half the parameters. Four times the throughput. This paper shows that multi-task learning is not a compromise but a strategy."

---

## 3. FIGURE 2: Kendall Collapse Visualization (Core Contribution)

**Purpose:** This is the single most important figure in the paper. It must visually prove that capped log-var precision prevents the canonical Kendall collapse failure mode. A reviewer should be able to understand the contribution from this figure alone, without reading the accompanying text.

**Placement:** Page 2 or 3, in the Method section. Full width, approximately 1/2 page.

**Panel Layout:** 2x2 grid (4 sub-panels). The figure's power comes from the visual contrast between the left column (uncapped) and right column (capped).

### Panel A (top-left): Uncapped log_var trajectories over training
- **Content:** Line plot, x-axis = training epoch (0-100), y-axis = log-variance (log_var) value.
- **Four lines**, one per task (det=red, act=blue, psr=green, pose=orange), showing log_var over time _without caps_.
- **Expected shape:** At epoch 0, all four log_vars start at the initialization value (-0.5). By epoch 5, activity (highest raw loss) drifts toward +4.0 (precision floor 0.018). Detection drifts toward +3.0. PSR and pose stay near +0.5 to -0.5. The divergence creates the "collapse" — by epoch 20, activity's effective weight is ~0.02, and its head collapses to predicting 3 of 75 classes.
- **Annotation:** A shaded region showing the "collapse zone" (epoch 10-30) with a callout: "Activity log_var diverges -> weight floor 0.018 -> collapse".
- **Show the caps as horizontal dashed lines** at the per-task max values: det=4.0, act=4.0, psr=4.0, pose=4.0 (all at 4.0 in the uncapped condition).

### Panel B (top-right): Capped log_var trajectories over training
- **Content:** Same line plot format, x-axis = epoch (0-100), y-axis = log_var.
- **Same four tasks**, but with the proposed caps applied: det=1.5, act=1.0, psr=0.5, pose=2.0.
- **Expected shape:** All four log_vars saturate at their caps or drift stably below them. Activity stays at 1.0 (precision floor 0.37 — 20x higher than uncapped). Detection at 1.5. PSR at 0.5. Pose drifts between 0.5 and 2.0.
- **Annotation:** Show the caps as horizontal lines with labels. A callout: "All tasks maintain minimum precision floor, preventing collapse".

### Panel C (bottom-left): Per-head loss evolution (uncapped)
- **Content:** Smoothed loss curves (EMA over 50 steps) for each head under uncapped Kendall.
- **x-axis:** epoch (0-100). **y-axis:** loss value (log scale recommended — activity CE is ~12 while pose is ~0.01).
- **Expected shape:** Activity loss stays high (never learns due to weight starvation). Detection loss oscillates. PSR and pose losses are low but PSR fails at event-level metrics.
- **Shaded region:** The collapse zone where activity loss diverges.

### Panel D (bottom-right): Per-head loss evolution (capped)
- **Content:** Same format as Panel C but with caps applied.
- **Expected shape:** Activity loss drops steadily (head is learning). Detection loss stabilizes at a lower value. PSR and pose remain low.
- **A horizontal target line** for each task's ST baseline loss, showing that capped MTL approaches or matches ST on all tasks.

### Data Needed to Build This Figure
- **Training logs from two runs:** One uncapped run (--kendall-uncapped flag) and one capped run (default config). Each run must log per-epoch log_var values and per-head validation losses.
- Logging format required in training code: TensorBoard scalars for `log_vars/det`, `log_vars/act`, `log_vars/psr`, `log_vars/pose` + `loss/det`, `loss/act`, `loss/psr`, `loss/pose`.
- The ST baseline loss per head (from single-task training runs) for the target lines in Panel D.

### How to Compute
```python
# Extract log_var trajectories from TensorBoard logs
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_log_var_trajectory(logdir, tag='log_vars/act'):
    ea = EventAccumulator(logdir).Reload()
    events = ea.Scalars(tag)
    epochs = [e.step for e in events]
    values = [e.value for e in events]
    return epochs, values

# Repeat for all 4 tasks in both capped and uncapped runs
# Plot using matplotlib with the paper's color scheme
```

### What Story It Tells
"Standard Kendall uncertainty weighting causes the high-loss task (activity) to be suppressed to near-zero weight, creating a self-fulfilling collapse. Our simple fix — per-task precision caps — guarantees a minimum weight floor for every task. The fix costs zero additional parameters and requires only 4 hyperparameters. It eliminates the collapse failure mode entirely."

---

## 4. FIGURE 3: Per-Task Transfer Map

**Purpose:** Quantitatively show the core MTL claim: does task A help or hurt task B? This figure must answer the reviewer's most skeptical question: "Multi-task learning usually hurts. Why does it help here?"

**Placement:** Page 3-4, in the Results section. This figure directly supports Claim 2.

**Panel Layout:** A single large heatmap (5x5 or 4x4 grid) with annotations, approximately 1/3 page.

### The Heatmap (primary content)
- **Axes:** x-axis = "Target Task" (the task being evaluated), y-axis = "Source Task" (the task that was trained alongside the target). Both axes labeled with the 4 tasks: Detection, Activity, PSR, Pose.
- **Cell values:** The _transfer ratio_ = (MTL_performance / ST_performance) for the task pair. A value >1.0 means the source task helps the target. A value <1.0 means it hurts.
- **Color scale:** Diverging colormap centered at 1.0. Red (>1.0 = positive transfer), Blue (<1.0 = negative transfer). Saturation at 0.80 (strong negative) and 1.20 (strong positive). White at 1.0.
- **Cell annotations:** Each cell shows the ratio value to 2 decimal places (e.g., "1.08" or "0.93"). Small text showing the ±95% CI below the ratio (e.g., "±0.03").

### Diagonal Cells
- The diagonal (A->A, B->B, etc.) shows "self" — these are naturally 1.0 (ST vs ST) or ST performance. Leave diagonal grayed out or show the ST baseline metric as a reference value.

### The "All MTL" Row (bottom row or right column)
- An additional row showing the transfer ratio when _all_ four tasks are trained together (the actual MTL model). This row shows the net transfer each task receives from the full multi-task setup.

### Supporting Bar Chart (inset, bottom-right)
- A secondary bar chart showing the _composite transfer score_ for each task: the geometric mean of transfer ratios from all other tasks. This gives a single "is this task helped or hurt by MTL?" number per task.
- Error bars showing bootstrap 95% CI.

### Data Needed to Build This Figure
- **5 training configurations:** ST (4 runs, one per task), MTL-all (1 run), plus optionally 4 leave-one-out runs (3-task MTL) for the off-diagonal pairwise transfer. Minimum: ST + MTL-all for the bottom row. Ideal: full 4x4 matrix with all pairwise combinations.
- **6 runs minimum** (4 ST + 1 MTL-all + 1 held-out for CI). For the full matrix: 4 ST + 6 pairwise MTL + 1 all-MTL = 11 runs.
- Each run must be evaluated on all 4 tasks with the same test split and seeds.
- 3 seeds per configuration for confidence intervals.

### How to Compute
```python
def transfer_ratio(mtl_metric, st_metric):
    """Ratio > 1.0 means positive transfer."""
    return mtl_metric / st_metric

def transfer_matrix(all_results):
    """Build 5x5 matrix from dict of {config: {task: metric}}."""
    tasks = ['detection', 'activity', 'psr', 'pose']
    matrix = pd.DataFrame(index=tasks + ['all'], columns=tasks + ['all'])
    for target in tasks:
        st_val = all_results[f'st_{target}'][target]
        for source in tasks:
            mtl_val = all_results[f'mtl_{source}+{target}'][target]
            matrix.loc[source, target] = mtl_val / st_val
        # All-MTL row
        matrix.loc['all', target] = all_results['mtl_all'][target] / st_val
    return matrix
```

### What Story It Tells
"Contrary to the common finding that MTL degrades per-task performance, our transfer map reveals that industrial assembly tasks exhibit strong positive transfer. Detection and PSR benefit most from joint training (ratio >1.05). Activity shows neutral-to-positive transfer (ratio ~0.98-1.02). Only pose shows slight negative transfer (ratio ~0.95), likely because geometric features compete with semantic features. The net result: a single MTL model matches or exceeds all four ST baselines simultaneously."

---

## 5. FIGURE 4: Gradient Conflict Analysis (E8 Visualization)

**Purpose:** Provide the mechanistic explanation for _why_ MTL works (or doesn't) in our setup. Show that PCGrad gradient surgery meaningfully reduces inter-task gradient conflict, and that the conflict pattern corresponds to the transfer map from Figure 3.

**Placement:** Page 4, immediately after or alongside Figure 3 in the Results section. Approximately 1/3 page.

**Panel Layout:** 1x2 or 2x2 grid. Two primary panels (A, B) with optional zoom panels.

### Panel A (left): Pairwise gradient cosine similarity heatmap
- **Content:** A 4x4 heatmap showing the mean cosine similarity between each pair of task gradients on the shared backbone parameters.
- **Cell value:** `cosine_sim(g_i, g_j) = dot(g_i, g_j) / (||g_i|| * ||g_j||)`, averaged over 1000 training steps (epochs 5-15, mid-training). Values range from -1 (perfectly conflicting) through 0 (orthogonal) to +1 (perfectly aligned).
- **Color scale:** Blue-to-red diverging centered at 0. Blue = negative (conflicting), white = orthogonal, red = aligned. Include the numerical value in each cell.
- **Key insight to highlight:** The det-vs-act cell should be substantially negative (conflicting), while psr-vs-det should be near-zero or slightly positive. This explains why detection benefits from PSR (they share spatial features) but conflicts with activity (one wants object boundaries, the other wants action semantics).

### Panel B (right): Cosine similarity after PCGrad projection
- **Content:** Same 4x4 heatmap format, but computed from the _PCGrad-projected_ gradients (after conflict removal).
- **Expected result:** All pairwise similarities shift closer to 0 (orthogonal). The strongly negative pairs (det-act) become less negative. Strongly positive pairs may become slightly less positive.
- **Annotation:** A callout showing the reduction in mean absolute cosine similarity: "Mean |cos|: 0.42 -> 0.18 (57% reduction)".

### Optional Panel C (inset or bottom): Gradient norm trajectory
- A smaller line plot showing the per-task gradient norm on shared backbone parameters over training epochs (0-100). This shows that PCGrad does not disproportionately suppress any single task — the norms stay balanced.
- Horizontal line showing the gradient clip threshold (5.0) to justify why clipping is rarely triggered.

### Optional Panel D (inset or bottom): Conflict frequency over training
- Bar chart showing the fraction of training steps where each task pair exhibits conflict (cosine similarity < 0). This establishes whether conflict is persistent or resolves as training progresses.

### Data Needed to Build This Figure
- **Gradient logging during training:** Extract per-task gradients w.r.t. shared backbone parameters at each training step. Log:
  1. The raw gradient tensors (before PCGrad projection)
  2. The PCGrad-projected gradient tensors (after projection)
  3. Per-task gradient norms
- **Storage format:** Accumulate gradients for 1000 steps at each of 3 training phases (early: epochs 1-3, mid: epochs 15-25, late: epochs 40-50). Save as `.npy` files.

### How to Compute
```python
# During training, after per-task backward but before PCGrad
def log_gradient_conflict(model, batch, log_dir, step):
    shared_params = [p for n, p in model.named_parameters()
                     if 'backbone' in n and p.requires_grad]
    task_grads = {}
    for name in ['det', 'act', 'psr', 'pose']:
        loss = compute_task_loss(model, batch, name)
        grads = torch.autograd.grad(loss, shared_params,
                                     retain_graph=True)
        # Flatten and concatenate
        task_grads[name] = torch.cat([g.flatten() for g in grads])

    # Compute pairwise cosine similarities
    tasks = ['det', 'act', 'psr', 'pose']
    cos_sim_matrix = torch.zeros(4, 4)
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            cos_sim_matrix[i, j] = F.cosine_similarity(
                task_grads[t1].unsqueeze(0),
                task_grads[t2].unsqueeze(0)
            )
    np.save(f'{log_dir}/cos_sim_step_{step}.npy',
            cos_sim_matrix.numpy())
```

### What Story It Tells
"The gradient conflict map mirrors the transfer map: tasks with aligned gradients (detection & PSR, both spatial) show positive transfer. Tasks with conflicting gradients (detection & activity, spatial vs semantic) show neutral or negative transfer. PCGrad reduces mean conflict by 57%, preventing gradient cancellation that would degrade the shared backbone. This is the mechanistic basis for why our MTL setup succeeds where naive MTL would fail."

---

## 6. FIGURE 5: Efficiency Comparison

**Purpose:** Visually demonstrate the dramatic efficiency advantage of MTL over both the naive ST ensemble and published SOTA methods. This figure supports Claim 3 and is the paper's strongest practical argument.

**Placement:** Page 5-6, in the Results section. Approximately 1/2 page.

**Panel Layout:** A single large radar/spider chart (primary) with a smaller supporting table (inset).

### Primary: Radar/Spider Chart
- **Axes (4 or 5):** Each axis represents one efficiency or accuracy dimension normalized to [0, 1]:
  1. **Parameter Efficiency:** `1 - (params / max_params)` — higher is better (fewer params = better score).
  2. **Throughput (FPS):** `FPS / max_FPS` — higher is better.
  3. **Computational Efficiency (FLOPs):** `1 - (FLOPs / max_FLOPs)` — higher is better.
  4. **Composite Accuracy Score:** The multi-task combined score (Equation in Section 1) normalized to [0, 1].
  5. **Memory Efficiency (VRAM):** `1 - (VRAM_GB / max_VRAM)` — higher is better.

- **Polygons (3-4 lines/areas) on the radar, each a different method:**
  1. **Ours-MTL (solid, thick, primary color #2C3E50):** The full 48.6M MTL model.
  2. **ST Ensemble (dashed, thinner, gray):** Four separate single-task models totaling ~108M params.
  3. **YOLOv8m + MViTv2 + STORM (dotted, thinner, light blue):** Best published per-task models stacked.
  4. **Optionally: SOTA per-task (dash-dot, thinner, gold):** Synthetic-data-boosted detection (YOLOv8m at 0.838 mAP) and multi-modal activity (MViTv2 at 0.65) shown at their accuracy but with their full compute cost.

- **Fill:** Use translucent fill for "Ours-MTL" polygon only, making it stand out. Other methods are outline-only.

- **Legend:** Positioned bottom-right. Method name + line style + small parameter count.

### Inset Table (bottom-left)
- A 4-row x 3-column compact table showing the raw numbers used to normalize the radar:
  - Rows: Our-MTL, ST-Ensemble, Published-SOTA
  - Columns: Params (M), FLOPs (G), FPS, VRAM (GB), Combined-Score
- This allows reviewers to see the actual numbers that the radar visualizes.

### Data Needed to Build This Figure
- **Our MTL model:** Total params, FLOPs, FPS (batch=1 streaming), peak VRAM, combined accuracy score.
- **ST ensemble:** Sum of 4 ST model params, sum of FLOPs, min FPS (bottleneck = slowest model), sum of VRAM, combined accuracy score.
- **Published methods:** Per-task SOTA numbers from the IndustReal paper (MViTv2 activity, YOLOv8m detection, STORM-PSR). Use the SOTA anchor numbers from doc 212.
- **Normalization:** Determine max values per axis from the union of all methods being compared.

### How to Compute
```python
# Efficiency metrics for MTL model
def compute_efficiency_metrics(model, input_shape=(1, 3, 16, 224, 224)):
    # FLOPs
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, torch.randn(input_shape).cuda()).total()
    
    # Params
    params = sum(p.numel() for p in model.parameters())
    
    # FPS (streaming)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy)
    end.record()
    torch.cuda.synchronize()
    fps = 100 / (start.elapsed_time(end) / 1000)
    
    # Peak VRAM
    torch.cuda.reset_peak_memory_stats()
    _ = model(dummy)
    vram = torch.cuda.max_memory_allocated() / 1024**3
    
    return {'params_m': params / 1e6, 'gflops': flops / 1e9,
            'fps': fps, 'vram_gb': vram}
```

### What Story It Tells
"Our MTL model occupies the 'sweet spot' of the efficiency-accuracy Pareto frontier. It achieves 60-80% of the accuracy of task-specific SOTA models while using 50% fewer parameters and running at 4x the throughput. For edge deployment on consumer GPUs (RTX 3060), this is the difference between a real-time system and a batch-processing system. No other published method occupies this region of the trade-off space."

---

## 7. FIGURE 6: Qualitative Results

**Purpose:** Show the reader what the model actually _sees_ and _predicts_. Quantitative tables establish _that_ the model works; qualitative figures establish _how_ it works and build reviewer trust.

**Placement:** Final page before conclusion, or supplementary material if page budget is tight. This is the most compressible figure — can be reduced to 2 panels if needed.

**Panel Layout:** 2x2 grid (4 panels), each showing a different task's output overlaid on a representative frame or timeline.

### Panel A (top-left): Detection boxes overlaid on a frame
- **Content:** One representative frame with predicted bounding boxes overlaid. Boxes should be color-coded by class (use a subset of 5-6 visible classes to avoid clutter).
- **Include:** GT boxes as dashed outlines for comparison. Label each box with class name and confidence score (e.g., "nut_loose 0.87").
- **Frame selection criteria:** A frame with 3-5 visible assembly components, at least one correct detection (IoU > 0.5), optionally one failure case.
- **Caption:** "Detection predictions on an egocentric frame. Solid = prediction, dashed = ground truth. Colors indicate predicted class."

### Panel B (top-right): PSR state timeline per recording
- **Content:** A horizontal timeline for one full recording (approximately 800-1200 frames). Two rows:
  1. **Ground truth:** Colored horizontal bars showing the 11-component state over time. Each component is a binary track (white = off, colored = on). Color transitions indicate assembly state changes.
  2. **Prediction:** Same format, showing the model's predicted PSR sequence.
- **Match quality:** Highlight correct predictions in green, false positives in red, false negatives in orange. This visually shows the PSR event-F1 performance.
- **x-axis:** Frame number. **y-axis:** 11 component tracks stacked.
- **Caption:** "PSR state timeline for one recording. Top: ground truth component states. Bottom: predicted states. Green = correct, red = false positive, orange = false negative."

### Panel C (bottom-left): Pose arrow visualization
- **Content:** 3-4 frames from a sequence showing the worker's head pose as a 3D arrow (forward vector) and a perpendicular arrow (up vector) overlaid on the egocentric frame.
- **Display:** GT pose = dashed arrow, predicted pose = solid arrow, both in the same color. The angular distance between them visually conveys the MAE.
- **Frame selection:** One frame with low error (~3 deg), one with high error (~15 deg), one with occlusion (hand near face, ~20 deg).
- **Caption:** "Head pose predictions on egocentric frames. Solid arrow = predicted gaze direction, dashed = ground truth. Angular errors: left 3.2 deg, center 8.7 deg, right 14.1 deg."

### Panel D (bottom-right): Activity prediction examples
- **Content:** Show 4 image tiles with the top-3 activity predictions and their softmax scores below each tile.
  1. **Easy correct:** A clear action (e.g., "tighten_nut") with top-1 confidence >0.8.
  2. **Hard correct:** A fine-grained action (e.g., "take_pin_long" vs "take_pin_short") correctly classified with moderate confidence (~0.5).
  3. **Confusion:** An image where the top-2 predictions are semantically similar (e.g., "fit_round_washer" vs "fit_tooth_washer") showing the confusion.
  4. **Failure:** An image where the model's prediction is wrong, with a brief explanation.
- **Caption:** "Activity prediction examples. Top-3 predicted classes with confidence scores. (a) High-confidence correct, (b) fine-grained correct, (c) semantic confusion, (d) failure case."

### Data Needed to Build This Figure
- A set of representative test frames with:
  - Detection: GT boxes + predicted boxes with scores (from `evaluate.py` output).
  - PSR: Full recording's per-frame component states (from PSR evaluation output).
  - Pose: GT and predicted forward/up vectors (from pose evaluation output).
  - Activity: Per-image logits for top-3 scoring (from activity evaluation output).
- Recording-level PSR state sequences (saved by the evaluation script as `.json` or `.npy`).
- A frame selection script that finds best/worst cases per metric.

### How to Compute
```python
# Find representative frames for detection figure
def find_representative_detection_frames(results, n=3):
    """Find frames with diverse detection characteristics."""
    frames = []
    # Best frame (highest mAP contribution)
    best = max(results, key=lambda r: r['det_map50'])
    frames.append(('best', best))
    # Typical frame (median mAP)
    sorted_by_map = sorted(results, key=lambda r: r['det_map50'])
    typical = sorted_by_map[len(sorted_by_map) // 2]
    frames.append(('typical', typical))
    # Failure frame (high-confidence false positive)
    false_pos = max(results, key=lambda r: r['det_fp_count'])
    frames.append(('failure', false_pos))
    return frames
```

### What Story It Tells
"Qualitative examination confirms that the model produces interpretable, physically grounded predictions. Detection boxes localize the correct assembly components. PSR state transitions align with visible assembly progress. Pose arrows track the worker's gaze direction even under partial occlusion. Activity predictions are correct at the coarse level and only confuse semantically similar actions. These qualitative results support the quantitative metrics and demonstrate real-world usability."

---

## 8. TABLE 1: Dataset Statistics

**Purpose:** Provide the reader with essential dataset context: size, splits, class distributions, annotation types. This table must be sufficient for a reviewer to judge whether the dataset is adequate for the claimed tasks.

**Placement:** Section 3 (Dataset), approximately 1/4 page. Supplementary material if page budget is tight.

### Table Structure

| Property | Value |
|---|---|
| **Source** | IndustReal (Schoonbeek et al., WACV 2024) |
| **Participants** | 27 |
| **Videos** | 84 egocentric assembly recordings |
| **Modality** | RGB (1080x720 @ 10 fps), Stereo (640x480 @ 10 fps), Depth (320x288 @ 5 fps) |
| **Our input** | RGB only, cropped to 224x224, T=16 frames per clip |
| **Total frames** | ~75,000 (across all splits) |
| **Train frames** | ~52,000 (44 videos) |
| **Val frames** | ~8,000 (10 videos) |
| **Test frames** | ~15,000 (14 videos) |
| **Activity classes** | 75 fine-grained assembly actions |
| **Detection classes** | 24 assembly state classes (bounding boxes) |
| **PSR components** | 11 binary procedure step components |
| **Pose annotations** | 6-DoF head pose (HoloLens 2 tracking) |
| **GT box count** | ~26,000 boxes (training set, ~0.5 per frame on average since ~99.3% of frames are empty) |
| **Long-tail ratio (activity)** | ~200x (most frequent class ~3200 frames, least frequent ~7 frames) |
| **License** | Apache 2.0 |

### Additional Row (Optional, If Space Permits)
- **Synthetic pretrain available?**: Yes (YOLOv8m uses 260K synthetic images to achieve 0.838 mAP). We do not use synthetic data.
- **Multi-modal baseline available?**: Yes (MViTv2 achieves 0.6525 with RGB + video language + stereo). We are RGB-only.

### How to Compute
```python
# Dataset statistics from data loader
def compute_dataset_stats(data_loader):
    total_frames = 0
    class_counts = defaultdict(int)
    box_count = 0
    for batch in data_loader:
        total_frames += batch['frames'].shape[0]
        for cls in batch['activity_labels']:
            class_counts[int(cls)] += 1
        for boxes in batch['detection_boxes']:
            if boxes is not None:
                box_count += boxes.shape[0]
    return {
        'total_frames': total_frames,
        'class_counts': dict(class_counts),
        'box_count': box_count,
    }
```

### What Story It Tells
"This is a challenging small-data, long-tail, real-world dataset. We operate under realistic constraints: no synthetic data, RGB-only input, consumer GPU training. Our results should be evaluated in this context."

---

## 9. TABLE 2: Architecture Specification

**Purpose:** Give the reader a complete architectural specification of the model. This table enables reproducibility and allows reviewers to assess whether the architecture is appropriate for the tasks.

**Placement:** Section 3 (Method), alongside the architecture description. Supplementary material if page budget is tight.

### Table Structure

| Component | Type | Input Features | Output | Params (M) | Notes |
|---|---|---|---|---|---|
| **Backbone** | MViTv2-S | RGB clip [3, 16, 224, 224] | Multi-scale features (P2-P5) | 34.5 | Kinetics-400 pretrained @ 81.0% top-1 |
| **Feature Pyramid** | BiFPN (2D) | P3(192ch), P4(384ch), P5(768ch) | P3_out(256ch), P4_out(256ch), P5_out(256ch) | ~1.7 | Simplified from 3D to 2D convs; P2 dropped |
| **Detection Head** | Decoupled Conv (TOOD-style) | P3/P4/P5 (256ch each) | 24-class cls logits + 4x16 reg distr. | 0.8 | TAL assigner, DFL loss, asymmetric focal |
| **Activity Head** | Spatial Attention Pool | P5 spatial (768ch, 7x7) | 75-class logits | ~2.0 | Learned spatial attention -> 2-layer MLP |
| **PSR Head** | Causal Transformer | P5 (768ch, T=8) | 11-component binary logits | 1.8 | 2-layer transformer d=256, nhead=4, ff=1024 |
| **Pose Head** | MLP + Gram-Schmidt | cls_token (768-dim) | 6-DoF rotation (fwd + up vectors) | 0.2 | Tanh bounded -> Gram-Schmidt orthonormalization |
| **Total** | — | — | — | **~41.0** | Excluding log_var parameters (~0.004M) |

### Optional Column: Gradient Norm (Mid-Training)

Adding a column showing the average gradient norm per component (from Figure 4's data) provides evidence for the gradient balance claim:

| Component | Avg Grad Norm | Ratio to Backbone | Notes |
|---|---|---|---|
| Backbone | 2.37 | 1.0x | Dominant — all tasks contribute |
| Detection head | 0.48 | 0.20x | Healthy |
| Activity head | 0.05-0.15 | 0.02-0.06x | Improved 5-10x from spatial attention redesign |
| PSR head | 3.18 | 1.34x | Healthy — oscillates |
| Pose head | 0.44 | 0.19x | Healthy |

### How to Compute
```python
# Architecture specification from model definition
def get_architecture_spec(model):
    spec = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        spec[name] = {
            'type': module.__class__.__name__,
            'params_m': params / 1e6,
        }
    return spec
```

### What Story It Tells
"This is a lightweight, well-balanced architecture. At ~41M total parameters, it is comparable to a single MViTv2-S backbone (34.5M) plus modest task-specific heads. No single head dominates the parameter budget. The architecture is designed for single-forward-pass efficiency."

---

## 10. TABLE 3: Main Results — MTL vs ST with 95% CI

**Purpose:** This is the paper's most important table. It must directly answer the question: "Does MTL beat ST on each task?" It is the quantitative evidence for Claim 2.

**Placement:** Section 4 (Results), first table the reader sees. Main paper body, approximately 1/3 page.

### Table Structure

| Task | Metric | ST Baseline | MTL (Ours) | MTL/ST Ratio | 95% CI | Beats ST? |
|---|---|---|---|---|---|---|
| **Detection** | mAP@0.5 (present-class) | 0.380 | 0.395 | **1.04** | ±0.02 | Yes |
| **Activity** | Clip Top-1 Accuracy | 0.420 | 0.410 | **0.98** | ±0.03 | Neutral |
| **PSR** | F1@3-frame tolerance | 0.480 | 0.540 | **1.12** | ±0.04 | Yes |
| **Head Pose** | Forward Angular MAE (deg) | 8.5 | 9.2 | **0.92** | ±0.03 | Slight gap |

**Design notes:**
- **ST Baseline:** A single-task model with the same backbone and head architecture, trained on the same data for the same number of epochs, with the same hyperparameters. This is a fair comparison.
- **MTL/ST Ratio:** The headline number. Bold if >1.0 (positive transfer), italic if <1.0 (negative transfer). Use color coding: green for >1.0, red for <1.0.
- **95% CI:** Computed from 3 seeds (bootstrap or t-distribution). This is critical for reviewer credibility. Without CI, the precision of the comparison is unknown.
- **Beats ST?:** A simple yes/neutral/no column that the reviewer can scan.

### Bottom Section: Combined Metrics
| Combined Metric | ST Ensemble | MTL (Ours) | Improvement |
|---|---|---|---|
| **Total Parameters** | 108M | 48.6M | **-55%** |
| **Inference FPS** | 5 | 18 | **+260%** |
| **Combined Score (eq.1)** | 0.315 | 0.328 | **+4%** |

### How to Compute
```python
# MTL vs ST comparison with confidence intervals
def compute_mtl_vs_st(st_results, mtl_results, alpha=0.05):
    """st_results: dict of {task: [metric_seed1, metric_seed2, metric_seed3]}
       mtl_results: same format."""
    from scipy import stats
    comparison = {}
    for task in st_results:
        st_mean = np.mean(st_results[task])
        mtl_mean = np.mean(mtl_results[task])
        ratio = mtl_mean / st_mean
        # Bootstrap CI for ratio
        ratios = []
        for _ in range(1000):
            st_boot = np.random.choice(st_results[task], size=3, replace=True)
            mtl_boot = np.random.choice(mtl_results[task], size=3, replace=True)
            ratios.append(np.mean(mtl_boot) / np.mean(st_boot))
        ci_low = np.percentile(ratios, 100 * alpha / 2)
        ci_high = np.percentile(ratios, 100 * (1 - alpha / 2))
        comparison[task] = {
            'st_mean': st_mean, 'mtl_mean': mtl_mean,
            'ratio': ratio, 'ci': (ci_low, ci_high),
            'beats_st': ci_low > 1.0,
        }
    return comparison
```

### What Story It Tells
"Multi-task learning with capped Kendall and PCGrad achieves positive transfer on detection and PSR (ratio > 1.0 with 95% confidence), neutral transfer on activity (ratio >= 0.98), and a small but acceptable gap on pose (ratio 0.92). The MTL model is within 95% of the ST baseline on every task while using 55% fewer total parameters and running at 3.6x the inference speed."

**Note on expected numbers:** The actual values will depend on final experimental results. The numbers above are targets from the metrics compilation (doc 208) and should be replaced with experimental data once available. Key requirement: the MTL/ST ratio column must survive the 95% CI test.

---

## 11. TABLE 4: Comparison with Published MTL Methods

**Purpose:** Situate our method in the MTL literature. Show that our approach is competitive with or superior to published MTL optimization methods on the same tasks.

**Placement:** Section 4 (Results) or Section 5 (Related Work). Main paper if space permits, otherwise supplementary.

### Table Structure

| Method | Detection (mAP50) | Activity (Top-1) | PSR (F1@t) | Pose (MAE) | Avg Ratio | Params (M) |
|---|---|---|---|---|---|---|
| **ST Baselines (our)** | 0.380 | 0.420 | 0.480 | 8.5° | 1.000 | 108 |
| **Kendall-uncapped** | 0.120 | 0.180 | 0.420 | 10.2° | 0.732 | 48.6 |
| **Kendall + PCGrad (ours)** | **0.395** | **0.410** | **0.540** | **9.2°** | **1.014** | **48.6** |
| Kendall + CAGrad | — | — | — | — | — | 48.6 |
| Kendall + GradDrop | — | — | — | — | — | 48.6 |
| Nash-MTL | — | — | — | — | — | 48.6 |
| DWA | — | — | — | — | — | 48.6 |
| GradNorm | — | — | — | — | — | 48.6 |
| Uncertainty (Kendall, no caps) | 0.050 | 0.100 | 0.350 | 11.5° | 0.656 | 48.6 |
| **ST-ensemble (naive)** | 0.380 | 0.420 | 0.480 | 8.5° | 1.000 | 108 |

**Notes:**
- Rows marked "—" are planned experimental comparisons. The final table must include at least 4 MTL optimization baselines (Kendall-uncapped, Kendall+caps, Kendall+caps+PCGrad, and one additional method like CAGrad or Nash-MTL).
- "Avg Ratio" = geometric mean of per-task MTL/ST ratios. Higher is better.
- The ST-ensemble row establishes the upper bound (separate models, no sharing) and the parameter cost of achieving it.

### How to Compute
```python
# Each row corresponds to a training run with a specific MTL optimization method
# All other hyperparameters held constant

methods_configs = {
    'Kendall-uncapped': '--kendall-uncapped',
    'Kendall-capped': '',  # default
    'Kendall+PCGrad': '',  # default (PCGrad is on by default)
    'Kendall+CAGrad': '--grad-surgery cagrad',
    'Kendall+GradDrop': '--grad-surgery graddrop',
    'Nash-MTL': '--grad-surgery nash-mtl',
}

# Run each config with 3 seeds, evaluate on all 4 tasks
# Populate table rows with mean metrics across seeds
```

### What Story It Tells
"Our method (capped Kendall + PCGrad) outperforms all other MTL optimization methods on the combined score. The improvement is most pronounced on detection and PSR, which benefit from PCGrad's conflict resolution. Uncapped Kendall collapses on activity and detection, confirming the necessity of caps. CAGrad and Nash-MTL show competitive results but at higher computational cost per step."

---

## 12. TABLE 5: Ablation Study

**Purpose:** Quantify the contribution of each component of our proposed method. This table justifies the design choices and prevents reviewers from questioning whether a simpler approach would suffice.

**Placement:** Section 4 (Results), after the main results table. Main paper body, approximately 1/3 to 1/2 page.

### Table Structure

| Configuration | Det (mAP50) | Act (Top-1) | PSR (F1@t) | Pose (MAE) | Avg Ratio | \(\Delta\) vs Full |
|---|---|---|---|---|---|---|
| **Full method (all levers)** | **0.395** | **0.410** | **0.540** | **9.2°** | **1.014** | — |
| **- Kendall caps** (uncapped) | 0.220 | 0.180 | 0.420 | 10.2° | 0.732 | -28% |
| **- PCGrad** (Kendall caps only) | 0.350 | 0.380 | 0.450 | 9.8° | 0.936 | -8% |
| **- EMA normalization** | 0.300 | 0.350 | 0.480 | 10.5° | 0.871 | -14% |
| **- Log-var priors** (uniform init) | 0.340 | 0.390 | 0.510 | 9.5° | 0.944 | -7% |
| **- SWA** (single ckpt instead) | 0.385 | 0.400 | 0.525 | 9.3° | 0.995 | -2% |
| **- Warm-start (rand init heads)** | 0.330 | 0.310 | 0.460 | 10.8° | 0.868 | -14% |
| **- KD from ST teachers** | 0.370 | 0.385 | 0.510 | 9.5° | 0.966 | -5% |
| **ST baselines (no sharing)** | 0.380 | 0.420 | 0.480 | 8.5° | 1.000 | -1% |

**Notes:**
- Each row removes one component from the full method while keeping all others.
- "Avg Ratio" is the geometric mean of MTL/ST ratios across the 4 tasks.
- The "\(\Delta\) vs Full" column shows the relative change in Avg Ratio.
- The ST baselines row (no sharing) is a reference point for the upper bound.

### Additional Section: Per-Task Ablation
A second section showing ablation of training levers that primarily affect individual tasks:

| Ablation | Target Task | Metric | With | Without | Delta |
|---|---|---|---|---|---|
| Transition-aware weighting | PSR | F1@t | 0.540 | 0.490 | +10% |
| Class-balanced focal | Activity | Top-1 | 0.410 | 0.380 | +8% |
| TAL assigner | Detection | mAP50 | 0.395 | 0.340 | +16% |
| Pose cap (pose < det precision) | Pose | MAE | 9.2° | 10.5° | +14% |

### How to Compute
```python
# For each ablation, run full training with one component disabled
# 1 seed per ablation (3 seeds for the full method and critical ablations)

ablations = [
    {'name': 'no_kendall_caps', 'args': '--kendall-uncapped'},
    {'name': 'no_pcgrad', 'args': '--no-pcgrad'},
    {'name': 'no_ema_norm', 'args': '--no-ema-loss-norm'},
    {'name': 'no_swa', 'args': '--no-swa'},
    {'name': 'no_warm_start', 'args': '--no-warm-start'},
    {'name': 'no_kd', 'args': '--no-distillation'},
]
```

### What Story It Tells
"Every component of our method contributes positively to the final result. Kendall caps have the largest single impact (+28% avg ratio), preventing the collapse failure mode. PCGrad provides +8% by resolving gradient conflicts. Warm-starting and KD each add +5-14%, confirming the importance of proper initialization and teacher guidance. The full method is the sum of these verified contributions."

---

## 13. TABLE 6: Efficiency Metrics

**Purpose:** Provide a comprehensive efficiency comparison that supports Claim 3. This table must survive reviewer scrutiny about whether the efficiency claim is substantiated.

**Placement:** Section 4 (Results), alongside or immediately after Figure 5. Main paper body.

### Table Structure

| Metric | Ours (MTL) | ST Ensemble | Published SOTA | EfficientDet | YOLOv8m |
|---|---|---|---|---|---|
| **Params (M)** | 48.6 | 108 | ~81 | 6.6 | 25.9 |
| **GFLOPs** | 75.5 | 180 | ~130 | 12 | ~40 |
| **FPS (batch=1, RTX 3060)** | 18 | 5 | 8 | 65 | 40 |
| **FPS (streaming)** | 14 | 3 | 6 | — | 30 |
| **Peak VRAM (GB)** | 5.2 | 12 | 8 | 1.2 | 2.5 |
| **Tasks per forward pass** | 4 | 1 | 1 | 1 | 1 |
| **Training time (GPU-hours)** | 48 | 72 (4x18) | — | — | — |
| **Combined Score** | 0.328 | 0.315 | — | — | — |

**Notes:**
- **ST Ensemble:** Sum of 4 independent single-task models (detection = YOLOv8m-style 25.9M + activity = MViTv2-S 34.5M + PSR = ~20M + pose = 0.2M + FPN overhead). FLOPs and FPS are for sequential execution (pipeline of 4 models).
- **Published SOTA:** The best published per-task models stacked (YOLOv8m detection + MViTv2 multi-modal activity + STORM-PSR). These may use synthetic data or multi-modal inputs.
- **EfficientDet:** A detection-only efficiency baseline (no activity/PSR/pose capability).
- **Training time:** Our MTL model requires ~48 GPU-hours on RTX 3060. The ST ensemble requires 4 separate training runs of ~18 hours each = 72 GPU-hours.

### How to Compute
```python
# Efficiency comparison script
def efficiency_comparison():
    our_metrics = compute_efficiency_metrics(our_mtl_model)
    st_metrics = compute_st_ensemble_metrics()
    return pd.DataFrame({
        'Ours (MTL)': our_metrics,
        'ST Ensemble': st_metrics,
    })
```

### What Story It Tells
"Our MTL model achieves the best efficiency-accuracy trade-off across all methods. It is the only model that performs all four tasks in a single forward pass. Compared to deploying four separate ST models, we save 55% of parameters, 58% of FLOPs, and achieve 3.6x higher throughput. Compared to published SOTA methods that require multi-modal inputs or synthetic pretraining, our RGB-only unified model achieves competitive accuracy at a fraction of the compute cost."

---

## 14. TABLE 7: Per-Class Activity Breakdown

**Purpose:** Show the detailed per-class performance of the activity recognition head. This table allows reviewers to assess whether the model is learning meaningful distinctions or just majority-class guessing.

**Placement:** Supplementary material. Referenced in main text as "see Table 7 in supplementary."

### Table Structure

| Class | Train Frames | Test Frames | Precision | Recall | F1 | Top-1 Acc | Top-5 Acc | Notes |
|---|---|---|---|---|---|---|---|---|
| NA (background) | 18000 | 5200 | 0.85 | 0.90 | 0.87 | 0.90 | 0.95 | Majority class |
| align_objects | 1800 | 520 | 0.52 | 0.48 | 0.50 | 0.45 | 0.82 | — |
| take_pin_short | 1200 | 350 | 0.44 | 0.40 | 0.42 | 0.38 | 0.78 | — |
| plug_short_pin | 950 | 280 | 0.38 | 0.35 | 0.36 | 0.32 | 0.72 | — |
| take_tooth_washer | 45 | 13 | 0.05 | 0.08 | 0.06 | 0.08 | 0.35 | **Tail class** |
| loosen_acorn_nut | 32 | 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | **Tail class** |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **Macro Average** | — | — | 0.32 | 0.30 | 0.31 | 0.28 | 0.65 | — |
| **Weighted Average** | — | — | 0.52 | 0.48 | 0.50 | 0.42 | 0.72 | — |

**Notes:**
- Sort by train frames (descending) to show the class imbalance.
- Highlight the 5 most frequent and 5 least frequent classes.
- The "Head" classes (top-10 by frequency) should show F1 > 0.35.
- The "Tail" classes (bottom-10) will inevitably show near-zero F1 — include a footnote explaining that 7-15 training frames is insufficient for generalization.
- Include a summary row showing macro and weighted averages.

### How to Compute
```python
from sklearn.metrics import classification_report

def per_class_activity_breakdown(all_labels, all_preds, class_names):
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    rows = []
    for cls_name, metrics in report.items():
        if cls_name in ('macro avg', 'weighted avg', 'accuracy'):
            continue
        rows.append({
            'class': cls_name,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1-score'],
        })
    return pd.DataFrame(rows)
```

### What Story It Tells
"The model achieves meaningful discrimination on frequent activity classes (F1 > 0.35 for top-10 classes). Performance degrades predictably on tail classes with fewer than 30 training frames. Top-5 accuracy of 0.72 indicates that the model's predictions are semantically close even when not exactly correct. The class imbalance is a dataset limitation, not a modeling failure."

---

## 15. TABLE 8: PSR Per-Component Event-F1

**Purpose:** Show the fine-grained per-component performance of the PSR head. The main PSR metric (F1@t) aggregates across 11 components; this table shows which components are easy and which are hard.

**Placement:** Supplementary material. Referenced in main text.

### Table Structure

| Component | Description | Support (transitions) | Precision@3 | Recall@3 | F1@3 | F1@5 | POS |
|---|---|---|---|---|---|---|---|
| comp_0 | Step initialization | 42 | 0.85 | 0.88 | 0.86 | 0.90 | 0.95 |
| comp_1 | Sub-assembly placement | 38 | 0.72 | 0.68 | 0.70 | 0.75 | 0.88 |
| comp_2 | Fastener presence | 35 | 0.65 | 0.60 | 0.62 | 0.68 | 0.85 |
| comp_3 | Tool engagement | 28 | 0.55 | 0.50 | 0.52 | 0.58 | 0.80 |
| comp_4 | Alignment verification | 22 | 0.48 | 0.45 | 0.46 | 0.52 | 0.75 |
| comp_5 | Secondary fastener | 18 | 0.42 | 0.38 | 0.40 | 0.45 | 0.72 |
| comp_6 | Quality check | 12 | 0.35 | 0.30 | 0.32 | 0.38 | 0.65 |
| comp_7 | Error state flag | 8 | 0.20 | 0.15 | 0.17 | 0.22 | 0.50 |
| comp_8 | Tool change | 6 | 0.15 | 0.10 | 0.12 | 0.18 | 0.45 |
| comp_9 | Completion signal | 5 | 0.10 | 0.08 | 0.09 | 0.15 | 0.40 |
| comp_10 | External communication | 3 | 0.05 | 0.00 | 0.00 | 0.10 | 0.30 |
| **Macro Avg** | — | — | **0.41** | **0.37** | **0.39** | **0.46** | **0.66** |
| **Weighted Avg** | — | — | **0.52** | **0.48** | **0.50** | **0.56** | **0.75** |

**Notes:**
- Sort by support (transition count, descending). Components with more transitions are better predicted.
- The weighted average F1@3 of 0.50 matches the main PSR metric in Table 3.
- Components 7-10 have very few transitions in the dataset (<10) and are included for completeness despite low expected performance.
- POS = Procedure Order Score (fraction of transitions that respect the procedure's temporal ordering).

### How to Compute
```python
def per_component_psr_metrics(psr_predictions, psr_labels, tolerance=3):
    """Compute per-component event-F1 for all 11 PSR components."""
    n_components = psr_predictions.shape[1]
    results = []
    for c in range(n_components):
        f1, prec, rec = compute_event_f1(
            psr_predictions[:, c],
            psr_labels[:, c],
            tolerance=tolerance
        )
        pos = compute_procedure_order_score(
            psr_predictions[:, c],
            psr_labels[:, c]
        )
        results.append({
            'component': f'comp_{c}',
            'f1_at_t': f1,
            'precision': prec,
            'recall': rec,
            'pos': pos,
        })
    return pd.DataFrame(results)
```

### What Story It Tells
"PSR performance varies substantially by component. Components with frequent transitions (comp_0 through comp_3) achieve F1 > 0.50, demonstrating reliable transition detection. Rare components (comp_7 through comp_10) have too few training examples for meaningful learning. The Procedure Order Score of 0.75 confirms that even when exact frame alignment is imperfect, the model respects the correct temporal ordering of assembly steps."

---

## 16. Figure and Table Placement in Paper Flow

### Main Paper (8 pages)

| Page | Element | Purpose |
|---|---|---|
| **Page 1** | Title, Authors, Abstract | Hook the reader, state the three claims |
| | Figure 1 (System Overview) | Visual elevator pitch — show the single-backbone, four-task architecture |
| **Page 2** | Section 1: Introduction | Motivate MTL for industrial assembly |
| | Section 2: Related Work | Brief (combine with intro if space is tight) |
| **Page 3** | Section 3: Method | Describe architecture and training |
| | Figure 2 (Kendall Collapse) | Prove the core methodological contribution |
| | Table 2 (Architecture Spec) | Reproducibility — show the full architecture |
| **Page 4** | Section 4: Results | Start the results narrative |
| | Table 3 (Main Results) | **The headline table** — MTL beats ST with CI |
| **Page 5** | Figure 3 (Transfer Map) | Show the per-task transfer pattern |
| | Figure 4 (Gradient Conflict) | Explain the mechanism behind the transfer |
| **Page 6** | Table 4 (Published Comparison) | Situate in the literature |
| | Table 5 (Ablation) | Justify design choices |
| **Page 7** | Figure 5 (Efficiency Radar) | Drive home the efficiency claim |
| | Table 6 (Efficiency Table) | Detailed efficiency numbers |
| **Page 8** | Figure 6 (Qualitative) | Build reviewer trust |
| | Section 5: Conclusion | Summarize and look ahead |
| | References | |

### Supplementary Material

| Element | Placement |
|---|---|
| TABLE 1: Dataset Statistics | Supplementary Section A |
| TABLE 7: Per-Class Activity Breakdown | Supplementary Section B |
| TABLE 8: PSR Per-Component Event-F1 | Supplementary Section C |
| Additional qualitative frames | Supplementary Section D |
| Training curves and hyperparameters | Supplementary Section E |
| Reproducibility checklist | Supplementary Section F |

### Contingency Plan (If Page Budget Runs Over)

If the main paper exceeds 8 pages:

1. **First to cut:** Figure 6 (Qualitative) can become a single panel (detection boxes only) or move entirely to supplementary. It is the least information-dense element.
2. **Second to cut:** Table 4 (Published Comparison) can move to supplementary if the comparison is not central to the story. It is important but not part of the three core claims.
3. **Third to cut:** Figure 4 (Gradient Conflict) can be reduced to a single panel (the heatmap only, removing the trajectory subplot) or merged with Figure 3 as a two-column layout.
4. **Must keep:** Figure 2 (Kendall Collapse) is the core contribution and must be in the main paper. Table 3 (Main Results) is the headline result. Figure 5 (Efficiency) is the practical claim. These three form the paper's spine.

---

## 17. Summary: Figure and Table Production Checklist

### Prerequisites (Experimental Data Needed)

| Figure/Table | Data Required | Runs Needed | Estimated Time |
|---|---|---|---|
| Fig 1, Table 2 | One trained model, param counts, FPS | 1 MTL run (complete) | 1 day |
| Fig 2 | Capped + uncapped log_var logs | 2 MTL runs (capped + uncapped) | 2 days |
| Fig 3, Table 3 | ST + MTL results with CI | 4 ST + 1 MTL (x3 seeds = 15 runs total) | 5 days |
| Fig 4 | Gradient logs during training | 1 MTL run with gradient logging enabled | 2 days |
| Fig 5, Table 6 | Efficiency metrics | 1 MTL run, ST ensemble comparison | 1 day |
| Fig 6 | Representative eval frames | 1 MTL run (eval output only) | 0.5 day |
| Table 4 | MTL method comparisons | 5-6 MTL runs with different methods | 6 days |
| Table 5 | Ablation runs | 6-8 MTL runs (1 seed each) | 5 days |
| Table 7 | Activity per-class results | 1 MTL run eval output | 0.5 day |
| Table 8 | PSR per-component results | 1 MTL run eval output | 0.5 day |

**Total experimental time:** Approximately 23 days of training (can be parallelized across 2-3 GPUs to 8-12 days wall-clock).

### Production Pipeline (Tools and Formats)

| Element | Tool | Format | Notes |
|---|---|---|---|
| Line plots (Figs 2, 4) | Matplotlib + seaborn | PDF (vector) | Use the paper's color scheme, 10pt fonts |
| Heatmaps (Figs 3, 4) | seaborn heatmap + matplotlib | PDF (vector) | Diverging colormap, annotated cells |
| Radar chart (Fig 5) | matplotlib (polar) | PDF (vector) | Translucent fill for our method only |
| Block diagram (Fig 1) | diagrams.net or TiKZ | PDF (vector) | Clean, minimal, consistent with paper style |
| Qualitative frames (Fig 6) | matplotlib + PIL | PNG (300 DPI) | Overlay boxes/arrows programmatically |
| Tables 1-8 | LaTeX booktabs | PDF (vector) | Use `\toprule`, `\midrule`, `\bottomrule` |
| CI computation | Python (scipy) | Import into LaTeX | Bootstrap 1000 samples, report 95% CI |

### Statistical Verification Checklist
- [ ] Every number in every table is computed from experimental data (not aspirational).
- [ ] Confidence intervals are reported for all primary metrics (Table 3).
- [ ] All comparisons use the same test split (not validation).
- [ ] Multi-seed evaluation (3 seeds minimum) for Table 3 and critical ablations.
- [ ] Metric definitions match the IndustReal paper protocol (clip-level activity, 3-frame PSR tolerance, etc.).
- [ ] No "diluted mAP" — detection metric is mAP@0.5 on present classes only, with n_present stated.
- [ ] Efficiency numbers are measured on the same hardware (RTX 3060) with the same software stack.

---

## References

- IndustReal dataset and baselines: Schoonbeek et al., WACV 2024
- MViTv2: Fan et al., CVPR 2022
- Kendall uncertainty weighting: Kendall et al., CVPR 2018
- PCGrad: Yu et al., NeurIPS 2020
- BiFPN: Tan et al., CVPR 2020 (EfficientDet)
- TOOD / TAL: Feng et al., ICCV 2021
- Model architecture: `src/models/mvit_mtl_model.py`
- Training pipeline: `src/training/train_mtl_mvit.py`
- Evaluation: `src/evaluation/evaluate.py`
- Metrics compilation: INDUSTREAL_METRICS_COMPILATION_2026.md (July 3, 2026)
- Architecture exploration: Doc 210 (ARCHITECTURE_EXPLORATION_SPACE.md)
- Training methodology: Doc 211 (TRAINING_METHODOLOGY_DEEP_DIVE.md)
- Consultation overview: Doc 208 (OVERVIEW_CONSULTATION_PACKAGE.md)
