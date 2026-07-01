# Figures Directory — ICHCIIS-26 Paper

Generate these figures before final submission:

| Figure | File | Source | Priority |
|--------|------|--------|----------|
| Fig 1: Architecture diagram | fig1_architecture.pdf | draw.io | HIGH |
| Fig 2: Detection confusion matrix | fig2_confusion.pdf | `evaluate.py` (24x24 matrix) | HIGH |
| Fig 3: Cost comparison bar chart | fig3_cost.pdf | matplotlib script | HIGH |

## Fig 1 — Architecture Diagram (draw.io suggested)

Elements:
- ConvNeXt-Tiny backbone → FPN
- 5 task heads branching off
- FiLM conditioning arrows (pose → C5, head pose → C5_mod)
- Label: "Single RTX 3060 ($299)"
- 720x1280 input → 5 outputs
- Clean, minimal, 3-4 colors max

## Fig 2 — Confusion Matrix

```bash
python3 src/evaluation/evaluate.py --ckpt src/runs/rf_stages/checkpoints/best.pth --split test
# Output: det_confusion_matrix.png in run directory
```

## Fig 3 — Cost Comparison

```python
import matplotlib.pyplot as plt
systems = ['POPW (Ours)', 'ViMAT', 'IFAS', 'Li et al.', 'Multi-Model']
costs = [799, 10000, 15000, 1000, 50000]
colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#f39c12', '#e74c3c']
plt.bar(systems, costs, color=colors)
plt.ylabel('3-Year Total Cost of Ownership (USD)')
plt.title('Hardware Cost Comparison')
plt.axhline(y=799, color='#2ecc71', linestyle='--', label='POPW: $799')
plt.legend()
plt.savefig('fig3_cost.pdf', dpi=300, bbox_inches='tight')
```
