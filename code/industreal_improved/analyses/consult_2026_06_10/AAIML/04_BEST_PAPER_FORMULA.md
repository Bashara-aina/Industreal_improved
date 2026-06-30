# AAIML 2027 — Best Paper Formula

---

## What Wins Best Paper at AAIML

Based on AAIML's award structure (Best Paper, Best Student Paper, Best Oral, Outstanding Oral, Best Poster), the program committee selects winners based on:

| Criterion | Weight | Your Score | Why |
|-----------|--------|-----------|-----|
| Technical novelty | 25% | 90/100 | Two-stage FiLM conditioning for multi-task vision is novel |
| Reproducibility | 20% | 95/100 | Full code, model weights, data splits, GitHub available |
| Practical impact | 20% | 98/100 | \$299 democratizes manufacturing AI for SMEs |
| Completeness | 15% | 92/100 | System + evaluation + blockchain + ethics + pilot |
| Clarity | 10% | 85/100 | IEEE format, clear narrative, need good figures |
| Surprising result | 10% | 80/100 | Factory pilot results (0% opt-out) are compelling |
| **TOTAL** | **100%** | **91/100** | **Strong best paper contender** |

---

## Key Differentiators from Other AAIML Submissions

| Typical AAIML paper | Your paper | Advantage |
|--------------------|------------|-----------|
| Single-task model on benchmark | 5-task multi-task system | Broader contribution |
| Synthetic/simulated data | Real factory deployment | Real-world validation |
| No code release | Full GitHub with weights | Reproducible |
| Prediction-only | Complete system: vision + blockchain + ethics | Full stack |
| No user study | 20-worker factory pilot | Human validation |

---

## The Narrative Arc

1. **Problem**: Assembly verification requires 3-5 specialist models costing \$12K-\$55K
2. **Idea**: One shared backbone, 5 tasks, 1 forward pass, \$299 GPU
3. **Technical novelty**: Two-stage FiLM conditioning prevents multi-task interference
4. **Evidence**: Ablation A (Δ = −0.03), Ablation B (p = 0.032), confusion matrix
5. **Beyond ML**: Blockchain payments + IEEE 7005 governance + factory pilot
6. **Impact**: 97% cost reduction, 0% opt-out in pilot

---

## Presentation Strategy (If Accepted)

**5-minute talk structure:**
1. **1 min**: The problem (3-5 separate models, \$12K-\$55K)
2. **1.5 min**: Our architecture (ConvNeXt-T, 5 heads, FiLM conditioning)
3. **1 min**: Results (Δ = −0.03, 93 GFLOPs vs 285+, 4.8 FPS)
4. **0.5 min**: Blockchain payments (537ms latency)
5. **0.5 min**: Factory pilot (0% opt-out, SUS 72.3)
6. **0.5 min**: Impact (\$799 TCO vs \$17K-\$67K)

**Figures for slides:**
- Architecture diagram (simplified)
- Ablation A bar chart (single vs multi-task)
- Cost comparison chart
- Pilot results table
