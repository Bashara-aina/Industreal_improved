# ICHCIIS-26 — Tables and Figures Pipeline

---

## Required Figures

| Figure | Description | Status | Tool | Priority |
|--------|-------------|--------|------|----------|
| Fig 1 | Architecture diagram | 🟡 Needs creation | draw.io / TikZ | HIGH |
| Fig 2 | 24x24 detection confusion matrix | 🟢 Already saved | evaluate.py | MEDIUM |
| Fig 3 | Cost comparison bar chart | 🟡 Needs creation | matplotlib | HIGH |
| Fig 4 | Head pose error distribution | 🟢 From logs | evaluate.py | LOW |
| Fig 5 | Ethical framework diagram | 🟡 Needs creation | draw.io | HIGH |
| Fig 6 | Blockchain payment flow | 🟡 Needs creation | draw.io | MEDIUM |

### Fig 1: Architecture Diagram (HIGH priority)
- ConvNeXt-Tiny backbone
- FPN neck
- 5 task heads branching off
- FiLM conditioning arrows
- Label: "Single forward pass on RTX 3060 ($299)"
- Style: clean, minimal, 3-4 colors max

### Fig 3: Cost Comparison (HIGH priority)
- Bar chart: POPW ($299) vs ViMAT (est. $10K) vs IFAS (est. $15K) vs Multi-model (est. $50K)
- Y-axis: Hardware cost (USD)
- Callout: "97% reduction"
- Style: Simple, readable, one strong accent color

### Fig 5: Ethical Framework Diagram (HIGH priority)
- IEEE 7005-2021 sections mapped to design principles
- 4 quadrants: Consent, Privacy, Transparency, Fairness
- Each with 2-3 implementation bullets
- Style: Clean infographic

---

## Required Tables

### Table 1: Competitor Analysis
| System | Tasks | GPU | Cost | Multi-task? |
|--------|-------|-----|------|-------------|
| ViMAT | Detection only | No spec | ~$10K | No |
| IFAS | Screw fastening only | Low-cost | ~$15K | No |
| Li et al. | Defect detection | GTX 3060 | ~$1K | Yes (2 tasks) |
| Multi-model | 3-5 specialists | Multi-GPU | $12K-$55K | Via separate models |
| **POPW (ours)** | **5 tasks** | **RTX 3060** | **$299** | **Yes (5 tasks)** |

### Table 2: Primary Results
| Task | Metric | Value |
|------|--------|-------|
| ASD detection | Present-class mAP50 | X.XX |
| ASD detection | mAP50 | X.XX |
| Head pose | Forward MAE | X.X° |
| Head pose | Up MAE | X.X° |
| Activity | Top-1 | X.X% |
| Activity | Top-5 | X.X% |
| PSR | F1(±3) | X.XX |
| Body pose | PCK@0.2 | X.XX |

### Table 3: Ablation A — Single-task vs Multi-task
| Configuration | det_mAP50_pc | Params | Forward passes |
|---------------|-------------|--------|----------------|
| Single-task (det only) | X.XX | 53M | 1 |
| Multi-task (all heads) | X.XX | 53M | 1 |
| Δ | ±X.XX | 0 | 0 |

### Table 4: Cost Analysis (3-year TCO)
| Item | POPW | Traditional Multi-Model |
|------|------|------------------------|
| GPU hardware | $299 | $12,000-$55,000 |
| Camera | $50 | $500-$2,000 |
| Installation | $200 | $2,000-$5,000 |
| Maintenance (3yr) | $250 | $2,500-$5,000 |
| **Total** | **$799** | **$17,000-$67,000** |

### Table 5: Ethical Framework Mapping (IEEE 7005-2021)
| IEEE Section | Design Principle | Implementation |
|-------------|-----------------|----------------|
| 5.1 Informed Consent | Opt-in/Opt-out | Supervisor sign-off alternative |
| 5.2 Data Governance | Local Processing | Edge GPU, no cloud, no face storage |
| 5.3 Transparency | Worker Dashboard | Real-time metrics, wallet notifications |
| 5.4 Fairness | Quality-weighted comp | No speed-only incentives |
| 5.5 Accountability | Audit Trail | Blockchain-immutable records |
| 5.6 Remediation | Grievance Process | Dispute resolution via smart contract |

---

## Figure Creation Scripts

### Confusion Matrix (already exists)
```bash
python3 src/evaluation/evaluate.py --ckpt src/runs/rf_stages/checkpoints/best.pth --split test
# Produces det_confusion_matrix.png in the run directory
```

### Cost Comparison (needs creation)
```python
import matplotlib.pyplot as plt
systems = ['POPW\n(Ours)', 'ViMAT', 'IFAS', 'Li et al.', 'Multi-Model\n(3 specialists)']
costs = [299, 10000, 15000, 1000, 50000]
colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#f39c12', '#e74c3c']
plt.bar(systems, costs, color=colors)
plt.ylabel('Hardware Cost (USD)')
plt.title('System Cost Comparison')
plt.axhline(y=299, color='#2ecc71', linestyle='--', label='POPW: $299')
plt.savefig('fig3_cost_comparison.png', dpi=300, bbox_inches='tight')
```

### Ethical Framework Diagram (needs creation)
Use draw.io or similar. Quadrant layout:
- Top-left: Consent → IEEE 5.1
- Top-right: Privacy → IEEE 5.2  
- Bottom-left: Transparency → IEEE 5.3
- Bottom-right: Fairness → IEEE 5.4
Center: IEEE 7005-2021
