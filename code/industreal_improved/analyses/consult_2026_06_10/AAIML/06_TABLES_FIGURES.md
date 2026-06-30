# AAIML 2027 -- Tables and Figures Needed with Creation Scripts

**Target**: 6-8 figures and 4-5 tables for the final submission. Every figure must be publication-ready (300 DPI, vector or high-res PNG, IEEE-compatible fonts).

---

## Figures Needed

### Figure 1: System Architecture Diagram (Full-Page Width)

**Purpose** (Section 3): Show the complete POPW pipeline from input frame to five task outputs with FiLM conditioning paths.

**Content**: Flowchart showing:
- Input RGB frame (720x1280)
- ConvNeXt-Tiny backbone -> FPN (P3-P7)
- Five task heads branching off:
  - Detection head (RetinaNet-style, 24 classes, boxes)
  - Body pose head (heatmap decoder, 17 keypoints)
  - Head pose head (MLP, 9-DoF)
  - Activity head (TCN + ViT, 74 classes)
  - PSR head (causal Transformer, 11 components)
- Two-stage FiLM conditioning: Body pose features modulate C5 via gamma/beta; HeadPoseFiLM applies second-stage modulation with stop-gradient
- Detection confidence pooled into activity head (dashed line)
- Output arrows: detection boxes, skeleton overlay, head orientation, activity label, step progress

**Critical detail**: Show which modules are frozen at which training stage (color-code RF1-RF4).

**Creation**: Drawio or TikZ. Source to be committed. Export as PDF vector.

**Script**: Design by hand in draw.io, export as PDF. No automated script.

---

### Figure 2: Detection Confusion Matrix (24x24 Heatmap)

**Purpose** (Section 4.2): Support the claim that 70% of errors are 1-bit Hamming-adjacent.

**Content**:
- 24x24 grid (true class vs predicted class)
- Color intensity = log(count+1) for visibility
- Diagonal highlighted
- Annotations: show Hamming distance categories in a sidebar or overlay
- Callout: highlight clusters of 1-bit-adjacent errors

**Creation script** (Python):

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load predictions and ground truth
# Assuming evaluate.py outputs: pred_classes, true_classes, hamming_distances
data = torch.load("results/confusion_data.pt")
cm = torch.zeros(24, 24, dtype=torch.long)
for t, p in zip(data["true"], data["pred"]):
    cm[t, p] += 1

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm.numpy(), annot=False, fmt="d", cmap="Blues",
            xticklabels=range(24), yticklabels=range(24), ax=ax)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Assembly State Detection Confusion Matrix (24 Classes)")

# Overlay Hamming distance categories
hamming = data.get("hamming_distances", None)
if hamming is not None:
    # Add inset or color bar showing Hamming distance info
    pass

fig.tight_layout()
fig.savefig("figures/confusion_matrix.pdf", dpi=300)
fig.savefig("figures/confusion_matrix.png", dpi=300)
print(f"1-bit-adjacent errors: {(hamming == 1).float().mean():.2%}")
```

---

### Figure 3: Ablation A -- Single-Task vs Multi-Task Comparison (Bar Chart)

**Purpose** (Section 4.4): Visualize the efficiency-accuracy tradeoff.

**Content**:
- Two groups: Single-task (detection only) vs Multi-task (det + pose + head-pose)
- Metric: present-class mAP50
- Error bars from bootstrap 95% CI [0.31, 0.37]
- Delta annotation: -0.03 (-8% relative)
- Inset or second panel: head pose angular error (Delta < 0.5 deg, show no degradation)

**Creation script** (Python):

```python
import matplotlib.pyplot as plt
import numpy as np

# Data (replace with three-seed values when available)
single_map = 0.37
multi_map = 0.34
ci_low, ci_high = 0.31, 0.37  # multi-task bootstrap CI

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Left: detection
bars = ax1.bar(["Single-Task", "Multi-Task"], [single_map, multi_map],
               color=["#4C72B0", "#DD8452"], width=0.5)
ax1.errorbar(1, multi_map, yerr=[[multi_map-ci_low], [ci_high-multi_map]],
             fmt='none', c='black', capsize=5)
ax1.set_ylabel("Present-class mAP50")
ax1.set_title("Detection Performance")
# Annotate delta
ax1.annotate(f"Delta = -0.03 (-8%)",
             xy=(1, multi_map), xytext=(0.5, multi_map - 0.08),
             arrowprops=dict(arrowstyle="->"), fontsize=9)

# Right: head pose
single_pose = 9.1  # placeholder
multi_pose = 9.0   # placeholder
ax2.bar(["Single-Task", "Multi-Task"], [single_pose, multi_pose],
        color=["#4C72B0", "#DD8452"], width=0.5)
ax2.set_ylabel("Forward MAE (degrees)")
ax2.set_title("Head Pose Estimation")

fig.tight_layout()
fig.savefig("figures/ablation_a.pdf", dpi=300)
```

---

### Figure 4: Ablation B -- FiLM Conditioning Effect (Bar Chart)

**Purpose** (Section 4.5): Show that cross-task FiLM conditioning improves activity recognition.

**Content**:
- Two bars: With FiLM (18.3%) vs Without FiLM (16.1%)
- Metric: Activity Top-1 accuracy
- P-value annotation: p = 0.032
- Bootstrap confidence intervals

**Creation script**: Similar to Figure 3. Single panel, two bars, p-value annotation.

---

### Figure 5: Cost Comparison Chart (Horizontal Bar)

**Purpose** (Section 1, Section 7): Dramatic visualization of 97% cost reduction.

**Content**:
- Horizontal bar chart comparing total system costs:
  - Traditional multi-model: $17,000-$67,000 (range bar)
  - ViMAT: $10,000+
  - IFAS: $15,000+
  - Li et al. (2 task): ~$1,000
  - POPW (5 task): $799 (3-year TCO) or $299 (single GPU)
- Color-code by number of tasks supported
- Annotation: "97% cost reduction" with arrow

**Creation script** (Python):

```python
import matplotlib.pyplot as plt

systems = ["Traditional\n(3-5 models)", "ViMAT\n(detection only)",
           "IFAS\n(screw only)", "Li et al.\n(2 tasks)", "POPW\n(5 tasks, ours)"]
# Lower and upper bounds for cost ranges
cost_low = [17000, 10000, 15000, 1000, 299]
cost_high = [67000, 15000, 20000, 1500, 799]
tasks = [1, 1, 1, 2, 5]

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = range(len(systems))

for i in range(len(systems)):
    color = "#DD8452" if i < 4 else "#4C72B0"
    ax.barh(y_pos[i], cost_high[i] - cost_low[i],
            left=cost_low[i], height=0.5, color=color, alpha=0.8)
    ax.text(cost_high[i] + 1000, y_pos[i], f"{tasks[i]} task(s)",
            va='center', fontsize=8)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(systems)
ax.set_xlabel("Cost (USD)")
ax.set_title("System Cost Comparison")
ax.set_xscale("symlog")  # log scale to show range
ax.axvline(x=299, color='green', linestyle='--', alpha=0.5, label="POPW GPU cost")
ax.legend()

fig.tight_layout()
fig.savefig("figures/cost_comparison.pdf", dpi=300)
```

---

### Figure 6: Pilot Results Dashboard (Combined Panel)

**Purpose** (Section 6): Show all pilot metrics in a single compelling figure.

**Content**: 2x2 or 1x3 panel:
- Panel A: NASA-TLX pre vs post (paired bar chart, p=0.04 annotation)
- Panel B: SUS score (72.3) with industry average benchmark line (68)
- Panel C: Trust (4.8/7) and Surveillance perception (2.3/7) -- two bars
- Panel D (optional): Radar chart with all metrics normalized

**Creation script** (Python):

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# Panel A: NASA-TLX
pre, post = 65.2, 58.4
pre_err, post_err = 12.1, 10.3
axes[0].bar(["Pre-Pilot", "Post-Pilot"], [pre, post],
            yerr=[pre_err, post_err], capsize=5,
            color=["#4C72B0", "#DD8452"])
axes[0].set_ylabel("NASA-TLX Score")
axes[0].set_title("Workload Reduction")
axes[0].annotate("p = 0.04", xy=(0.5, 75), ha='center', fontsize=9)

# Panel B: SUS
sus, sus_err = 72.3, 8.9
axes[1].bar(["POPW SUS"], [sus], yerr=[sus_err], capsize=5, color="#4C72B0")
axes[1].axhline(y=68, color='red', linestyle='--', label="Industry avg (68)")
axes[1].legend(fontsize=8)
axes[1].set_ylabel("SUS Score")
axes[1].set_title("Usability")

# Panel C: Trust and Surveillance
trust, surv = 4.8, 2.3
trust_err, surv_err = 1.2, 1.4
axes[2].bar(["Trust (1-7)", "Surveillance\nPerception (1-7)"],
            [trust, surv],
            yerr=[trust_err, surv_err], capsize=5,
            color=["#4C72B0", "#DD8452"])
axes[2].set_ylabel("Score")
axes[2].set_title("Attitudes")

fig.tight_layout()
fig.savefig("figures/pilot_dashboard.pdf", dpi=300)
```

---

### Figure 7: Training Loss and Kendall Weight Evolution (Multi-Panel)

**Purpose** (Section 3.5): Show the staged training protocol working in practice.

**Content**:
- Panel A: Training loss curves for each stage (RF1-RF4), 4 colors
- Panel B: Kendall homoscedastic uncertainty weights (log(sigma^2)) over training steps for each task head
- Panel C: Task-specific validation metrics at each stage boundary

**Creation script**: Extract from training logs (wandb or tensorboard). Plot using matplotlib.

---

### Figure 8: Blockchain Micropayment Pipeline (Diagram)

**Purpose** (Section 5): Explain the 4-step x402 flow visually.

**Content**: Swimlane diagram with 4 columns:
1. Worker Device (camera + POPW inference)
2. Local Verifier (PSR triggers on step completion)
3. Solana Network (x402 transaction)
4. Worker Wallet (balance update)

Arrows: Step detected -> hash generated -> transaction submitted -> confirmations -> notification.

**Creation**: Drawio or sequence diagram in TikZ.

---

## Tables Needed (in the .tex file)

### Table 1: Competitor Analysis (already in paper, verify numbers)
Columns: System | Tasks | GPU Cost | Multi-task? | Ethics? | Pilot?

### Table 2: Primary Results (already in paper)
Add 3-seed variance when available. Add "Bootstrap 95% CI" column.

### Table 3: Staged Training Protocol (already in paper, verify)

### Table 4: Ablation Results (new or merge into primary)
Proposed layout:
| Ablation | Variant | Det mAP50 | Pose MAE | Activity Top-1 |
|----------|---------|-----------|----------|-----------------|
| Full POPW (multi-task) | With FiLM | 0.34 | 9.1 | 18.3% |
| Single-task baseline | - | 0.37 | <0.5 deg diff | - |
| No FiLM conditioning | No PoseFiLM | - | - | 16.1% |
| Detection + Pose only | No activity | 0.35 | 9.0 | - |

### Table 5: Efficiency Benchmarking (new)
| Metric | POPW (5 tasks) | YOLOv8m (1 task) | MViTv2-S (1 task) | STORM-PSR (1 task) | 3-model total |
|--------|---------------|------------------|-------------------|--------------------|---------------|
| Params (M) | 53.0 | 25.9 | 36.0 | 28.4 | 90.3 |
| GFLOPs | 93 | 79 | 170 | 112 | 361 |
| FPS on RTX 3060 | 4.8 | - | - | - | - |
| GPU required | 1x $299 | 1x $299 | 1x $2K+ | 1x $299 | 3 GPUs |
| Tasks covered | 5 | 1 | 1 | 1 | 3 |

### Table 6: Pilot Results (already in paper, verify)
Add effect sizes (Cohen's d for NASA-TLX).

### Table 7: IEEE 7005 Mapping (already in paper, verify)
Mark (P) items as "in development" rather than just "design principle."

---

## Figure Generation Pipeline

Create `scripts/generate_figures.py` that produces all figures in batch:

```python
#!/usr/bin/env python3
"""Generate all figures for AAIML 2027 submission."""
import subprocess
from pathlib import Path

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

scripts = [
    "fig_confusion_matrix.py",
    "fig_ablation_a.py",
    "fig_ablation_b.py",
    "fig_cost_comparison.py",
    "fig_pilot_dashboard.py",
    "fig_training_curves.py",
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", f"scripts/{script}"], check=True)

print("All figures generated in figures/")
```

---

## Pre-Submission Figure Checklist

- [ ] Figure 1: Architecture diagram -- vector PDF, font sizes readable at 2-col width
- [ ] Figure 2: Confusion matrix -- 300 DPI, class labels visible
- [ ] Figure 3: Ablation A -- colorblind-friendly, delta annotation clear
- [ ] Figure 4: Ablation B -- p-value visible, error bars
- [ ] Figure 5: Cost comparison -- log scale readable, 97% annotation
- [ ] Figure 6: Pilot dashboard -- all three panels aligned, legend readable
- [ ] Figure 7: Training curves -- stage boundaries marked
- [ ] Figure 8: Blockchain pipeline -- swimlane format, clear
- [ ] All scripts committed to `scripts/` directory
- [ ] All figures exported as PDF (vector) AND PNG (for arXiv fallback)
- [ ] Figure captions written and match IEEE format
