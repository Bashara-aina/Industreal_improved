# Analysis & Visualization

Scripts that generate plots, tables, and reports from training logs and
checkpoints.

## Scripts

- `figure_gradient_norms.py` — Per-head backbone gradient norm bar chart
- `figure_logvar_trajectories.py` — Kendall log-var trajectories
- `figure_training_curves.py` — Per-task metric vs. epoch training curves
- `plot_training_curves.py` — Plot training curves from log file
- `activity_confusion_matrix.py` — Activity recognition confusion matrix
- `build_mtl_st_comparison_table.py` — MTL vs ST comparison table
- `build_soup.py` — Build model soup (averaged weights from multiple runs)
- `compare_checkpoints.py` — Compare two checkpoints (parameter diff, metric diff)
- `calibrate_det_threshold.py` — Detection score-threshold calibration
- `generate_paper_table.py` — Paper headline table generator
- `measure_efficiency.py` — Efficiency metrics (FPS, params, GFLOPs)

## Usage

```bash
# Generate paper table from a directory of checkpoints
python scripts/analysis/generate_paper_table.py \
    --checkpoint-dir runs/paper_run/

# Plot training curves
python scripts/analysis/plot_training_curves.py \
    --log-file runs/mtl_v3.7/train.log \
    --output /tmp/training_curves.png

# Compare two checkpoints
python scripts/analysis/compare_checkpoints.py \
    runs/mtl_v3.5/checkpoints/phase2_e5_b0.pth \
    runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth
```