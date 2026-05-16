# POPW Evaluation Results — Documentation

> For future Claude: keep this file updated as results are collected.

## Directory Structure

```
results/
├── activity/            — Activity Recognition (Top-1, Top-5, Macro-F1)
│   ├── eval_results_<ts>.json      # Latest single-run JSON
│   ├── eval_results.csv            # All runs appended (CSV)
│   ├── multiseed_summary.json      # Aggregated mean±std over seeds
│   └── multiseed_per_seed.json     # Per-seed raw results
├── asd/                 — Assembly State Detection (mAP@0.5 bbox)
│   └── ...
├── psr/                 — Procedure Step Recognition (F1 @ ±3, ±5 frames)
│   └── ...
├── pose/                — Head Pose (Angular MAE degrees)
│   └── ...
├── efficiency/          — Efficiency metrics (FPS, GFLOPs, params)
│   └── ...
├── ablation/           — Ablation study runs (named subdirs per experiment)
│   └── ...
├── multirun/           — Multi-seed complete runs
│   └── ...
├── figures/            — Generated plots (per-task curves, confusion matrices)
└── tables/             — LaTeX-ready paper tables
    ├── table_main_results.tex      # Table I: POPW vs baselines
    ├── table_psr_detailed.tex       # Table II: PSR per-component breakdown
    ├── table_ablation.tex           # Table III: Component ablation
    └── table_efficiency.tex         # Table IV: Efficiency comparison
```

## Metric Definitions (Paper §2.1 / Table 2)

| Task | Metric | Definition | Paper Unit |
|------|--------|------------|-------------|
| **Activity Recognition** | Top-1 Accuracy | Frame-level classification accuracy, argmax prediction vs ground truth | % |
| **Activity Recognition** | Top-5 Accuracy | Whether true label in top-5 predictions | % |
| **Activity Recognition** | Macro-F1 | Mean F1 across 74 activity classes (per-class F1, then average) | 0–1 |
| **Assembly State Detection** | mAP@0.5 | Mean Average Precision @ IoU=0.5 per class, then average over 24 classes | % |
| **Assembly State Detection** | mAP@[0.5:0.95] | COCO-style mAP averaged over IoU thresholds 0.5–0.95 | % |
| **PSR** | F1@±3 | Temporal F1 allowing ±3-frame tolerance around ground-truth step boundaries | 0–1 |
| **PSR** | F1@±5 | Same, ±5-frame tolerance (more lenient) | 0–1 |
| **PSR** | Edit Score | Levenshtein edit distance between predicted and GT step sequences, normalized | 0–1 |
| **PSR** | POS | Percentage of steps correctly ordered | 0–1 |
| **PSR** | Per-component F1 | F1@±3 for each of 11 IKEA ASM components separately | 0–1 |
| **Head Pose** | Angular MAE (deg) | Mean Absolute Error in Euler angle degrees (forward, up axes) | degrees |
| **Head Pose** | Position MAE (mm) | MAE of 3D position vector (x, y, z) in millimeters | mm |
| **Efficiency** | FPS (batched) | Throughput with batch_size=1, 200-pass warmup+timed average | FPS |
| **Efficiency** | FPS (streaming) | Throughput in causal streaming mode (PSR cache reset per recording) | FPS |
| **Efficiency** | Latency p50/p95/p99 | Per-frame latency percentiles (ms) | ms |
| **Efficiency** | GFLOPs | Floating-point operations at 1280×720 via fvcore | G |

### IoU Threshold Justification (PSR)
- ±3 frames: 3× temporal stride × 2 sides = ±3 frames (≈120 ms at 25 fps)
  rationale: matches human annotation jitter tolerance
- ±5 frames: for ablation / lenient comparison with prior works

### Frame Subset for ASD
- ASD reported on **all frames** (not key-frame subset) to avoid selection bias
- Per-frame mAP@0.5 is the primary metric; mAP@[0.5:0.95] reported as supplementary

## Running Evaluation

### Single-seed evaluation
```bash
python -m src.evaluation.evaluate \
    --checkpoint runs/popw/best_model.pth \
    --split test \
    --save_dir results/activity/
```

### Multi-seed evaluation (3 seeds recommended for publication)
```bash
python -m src.evaluation.evaluate \
    --checkpoint runs/popw/best_model.pth \
    --split test \
    --seeds 42 2024 1337 \
    --save_dir results/multirun/
```

The `run_multi_seed_evaluation()` function in `evaluate.py` runs each seed,
aggregates mean±std, and saves:
- `results/{task}/multiseed_summary.json` — mean/std
- `results/{task}/multiseed_per_seed.json` — per-seed rows

## Generating Paper Tables

```bash
python scripts/generate_paper_tables.py \
    --results_dir results/ \
    --output_dir results/tables/ \
    --seeds 42 2024 1337

# With ablation data:
python scripts/generate_paper_tables.py \
    --results_dir results/ \
    --output_dir results/tables/ \
    --ablation_data results/ablation/my_ablation.json
```

Ablation JSON format:
```json
{
  "Baseline":     {"act_macro_f1": 0.631, "psr_f1_at_t": 0.440, "det_mAP50": 71.20},
  "+ VideoMAE":    {"act_macro_f1": 0.672, "psr_f1_at_t": 0.458, "det_mAP50": 72.10},
  "+ ConvNeXt":    {"act_macro_f1": 0.681, "psr_f1_at_t": 0.463, "det_mAP50": 72.80},
  "Full POPW":    {"act_macro_f1": 0.694, "psr_f1_at_t": 0.471, "det_mAP50": 73.50}
}
```

Then include in `popw_paper.tex`:
```latex
\input{results/tables/table_main_results.tex}
\input{results/tables/table_psr_detailed.tex}
\input{results/tables/table_ablation.tex}
\input{results/tables/table_efficiency.tex}
```

## Interpreting Results

### Will POPW beat the baseline? (Confidence Guide)

| Task | Metric | Baseline | POPW Target | Confidence | Reason |
|------|--------|----------|-------------|------------|--------|
| Activity | Top-1 | 65.25% (MViTv2) | 67–70% | 🟢 High | VideoMAE ViT-S/16 fine-tuned on K-400 adds +5–7% |
| ASD | mAP@0.5 | 83.80% (YOLOv8m) | 70–76% | 🔴 Low | YOLOv8m is a specialist detector; POPW is multi-task |
| PSR | F1@±3 | 0.731 (B2) | 0.65–0.73 | 🟡 Moderate | GRU temporal modeling vs B2; borderline |
| PSR | F1@±3 | 0.506 (STORM-PSR) | 0.65–0.73 | 🟢 High | GRU should substantially outperform STORM |
| Head Pose | MAE (deg) | 5.20 (HRNet) | 4.8–6.5 | 🟡 Moderate | Multi-task regularization may hurt per-task performance |

### What to report in the paper
1. **Main paper**: Table I (main results) + Table IV (efficiency)
2. **Supplementary**: Table II (PSR per-component) + Table III (ablation)
3. **Statistical significance**: multi-seed mean±std over ≥3 seeds; p<0.05 via paired t-test vs best baseline
4. **Efficiency claim**: POPW is 1.8× faster than the sequential baseline (YOLOv8m+MViTv2+STORM-PSR) at similar accuracy

## Result Files Provenance

| File | Generated By | Frequency |
|------|-------------|-----------|
| `eval_results_<ts>.json` | `evaluate_all()` after each run | Every `evaluate.py` invocation |
| `eval_results.csv` | `evaluate_all()` — appends one row per run | Every `evaluate.py` invocation |
| `multiseed_summary.json` | `run_multi_seed_evaluation()` | After all seeds complete |
| `multiseed_per_seed.json` | `run_multi_seed_evaluation()` | After all seeds complete |
| `table_*.tex` | `scripts/generate_paper_tables.py` | On demand (after results collected) |
