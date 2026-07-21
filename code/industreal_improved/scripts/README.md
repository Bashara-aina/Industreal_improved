# Scripts Directory — Organization

This directory contains all utility scripts for training, evaluation, analysis,
and verification of the multi-task IndustReal model. Scripts are organized by
purpose into subdirectories.

## Quick Reference

### 📊 `eval/` — Evaluation scripts
Run these on trained checkpoints to measure performance on each task.

| Script | Purpose |
|--------|---------|
| `eval_all_heads.py` | **Comprehensive 4-head evaluation** (Detection mAP, Activity top-1/5, Pose MAE, PSR F1) |
| `eval_mvit_mAP.py` | Detection mAP@0.5 (uses training-correct decode formula) |
| `eval_activity_75class.py` | Activity Recognition top-1/5 accuracy (75-class) |
| `eval_pose_norm_fix.py` | Head Pose angular MAE (degrees) |
| `eval_psr_transition_f1.py` | PSR transition event F1 score |
| `eval_v38_fix.py` | Compare v3.8_fix vs v3.7 baseline |
| `eval_detection_dual_protocol.py` | Detection with COCO + custom protocols |
| `eval_with_tta.py` | Detection with horizontal-flip TTA |
| `eval_test_split.py` | Evaluate on held-out test split |
| `eval_checkpoint.py` | Generic checkpoint evaluation (legacy POPWMultiTaskModel) |

**Recommended:** Start with `eval_all_heads.py` for full audit of all 4 heads.

### 🚂 `train/` — Training scripts
Launch training runs.

| Script | Purpose |
|--------|---------|
| `train_mtl_mvit.py` | **Main MTL training script** (full multimodal, 4 heads) |
| `train_v8_multitask.py` | V8 multi-task pipeline |
| `train_st_act.py` / `train_st.py` | Single-task baselines |
| `decoupled_act_retrain.py` | Decoupled activity classifier retrain |
| `train_psr_repair_wrapper.py` | PSR head repair with bf16 mixed precision |
| `st_act_standalone.py` | Single-task activity training (no DataLoader) |
| `*.sh` | Launcher scripts for various training configs |

### 🔬 `probes/` — Diagnostic probes
Quick experiments to test hypotheses about training dynamics.

| Script | Purpose |
|--------|---------|
| `probe_logit_bias_disable.py` | **Test if disabling update_logit_bias() helps BG suppression** |
| `tal_probe_correct.py` | **Corrected TAL vs 3×3 probe** (uses correct dataset) |
| `tal_probe_fixed.py` | Original TAL probe (USES WRONG DATASET — verdict meaningless) |
| `overfit_probe.py` | Overfit probe — most important diagnostic |
| `overfit_50img_cls.py` | Overfit 50 images for classification |
| `mvp_probe3_psr_ab.py` | PSR temporal-resolution A/B |
| `mvp_probe4_tal_vs_3x3.py` | Detection TAL vs 3×3 probe |
| `debug_q43.py` / `debug_q43_v2.py` | Debug Q43 canonical POS |
| `e8_gradient_diagnostic.py` / `..._lite.py` | E8 gradient-flow diagnostic |
| `check_weight_evolution.py` | Check weight evolution during training |
| `check_train_val_subject_disjoint.py` | Verify subject disjointness |

### 📈 `analysis/` — Analysis & visualization
Generate plots, tables, and reports from training logs/checkpoints.

| Script | Purpose |
|--------|---------|
| `figure_gradient_norms.py` | Per-head backbone gradient norm bar chart |
| `figure_logvar_trajectories.py` | Kendall log-var trajectories |
| `figure_training_curves.py` | Per-task metric vs. epoch curves |
| `plot_training_curves.py` | Plot training curves from log |
| `activity_confusion_matrix.py` | Activity confusion matrix |
| `build_mtl_st_comparison_table.py` | MTL vs ST comparison table |
| `build_soup.py` | Build model soup (averaged weights) |
| `compare_checkpoints.py` | Compare two checkpoints (param diff) |
| `calibrate_det_threshold.py` | Detection score-threshold calibration |
| `generate_paper_table.py` | Paper headline table generator |
| `measure_efficiency.py` | Efficiency metrics (FPS, params, GFLOPs) |

### ✅ `verify/` — Verification scripts
Sanity checks and validation tests.

| Script | Purpose |
|--------|---------|
| `verify_checkpoint.py` | Verify checkpoint loads + 4 heads produce outputs |
| `verify_act_grouping.py` | Verify activity grouping against class names |
| `verify_gt_coordinates.py` | Verify ground truth coordinates |
| `verify_subject_split.py` | Verify train/val/test subject splits |
| `discover_test_subjects.py` | Discover test subjects in dataset |
| `integration_test.py` | Full train_step integration test |
| `test_anchor_normalization.py` | Test anchor normalization |

### 🔧 `utils/` — Utilities
General-purpose tools and helpers.

| Script | Purpose |
|--------|---------|
| `export_onnx.py` | Export model to ONNX |
| `profile_dataloader.py` | Profile DataLoader throughput |
| `mediapipe_pose_baseline.py` | MediaPipe pose baseline (reference) |
| `training_monitor.py` | Detect training anomalies |
| `mvp_smoke_suite.py` | MVP smoke suite (diagnose 0.0/0.008 numbers) |
| `minimal_smoke_test.py` | Minimal smoke test |
| `smoke_test.py` | Full smoke test |
| `smoke_test_4heads.py` | 4-head smoke test |
| `run_q43_canonical_pos.py` | Q43 canonical-order POS baseline |
| `kill_training.sh` | Kill running training |
| `monitor_training.sh` | Monitor training progress |
| `freeze_checkpoint.sh` | Freeze a checkpoint |
| `launch_*.sh` | Training launchers |
| `restart_*.sh` | Training restart helpers |
| `full_pipeline_v1.sh` | Full v1 pipeline |
| `reproduce.sh` | Reproducibility script |
| `download_yolov8m_industreal.sh` | Download YOLOv8 weights |
| `training_pids.txt` | Last training PIDs |

### 🎨 `visualization/` — Visualization scripts
Visual debugging of model outputs.

| Script | Purpose |
|--------|---------|
| `diagnostic_visualization.py` | Diagnostic plots |
| `visualize_filtered.py` | Visualize filtered predictions |
| `visualize_hand_joints.py` / `..._v2.py` / `..._v3.py` | Hand joint visualization |
| `visualize_head_pose.py` | Head pose visualization |
| `visualize_psr_transitions.py` | PSR transition visualization |
| `visualize_with_markers.py` | Visualization with markers |

### 📚 `training/` — Training sub-scripts
Sub-scripts used by main training scripts.

| Script | Purpose |
|--------|---------|
| `calibrate_anchors.py` | Anchor calibration utilities |
| `cross_validate.py` | Cross-validation helpers |
| `efficiency_report.py` | Efficiency reporting |
| `generate_ablation_table.py` | Ablation study table |
| `generate_paper_tables.py` | Paper tables generator |
| `run_multi_seed.py` | Multi-seed training |
| `test_e2e_training.py` | End-to-end training test |

## Naming Conventions

- `eval_*` — Evaluation scripts (output metrics, may be slow)
- `train_*` — Training scripts (modify checkpoints, take hours)
- `probe_*` — Quick diagnostic experiments (run in minutes)
- `mvp_probe*` — MVP probes (specific questions)
- `figure_*` / `plot_*` — Visualization
- `verify_*` / `check_*` / `test_*` — Sanity checks
- `debug_*` — Targeted debugging
- `*.sh` — Bash launcher/automation scripts

## Common Workflows

### 1. Train a model
```bash
bash scripts/train/run_*.sh  # Use a launcher script
# OR
python scripts/train/train_mtl_mvit.py [args]
```

### 2. Evaluate a checkpoint on all 4 heads
```bash
python scripts/eval/eval_all_heads.py \
    --checkpoint runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
    --output /tmp/eval.json
```

### 3. Quick detection-only mAP
```bash
python scripts/eval/eval_mvit_mAP.py \
    --checkpoint runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
    --max-frames 2000
```

### 4. Test a hypothesis
```bash
python scripts/probes/probe_logit_bias_disable.py --n-steps 500
```

### 5. Generate paper tables
```bash
python scripts/analysis/generate_paper_table.py \
    --checkpoint-dir runs/paper_run/
```

## Deprecated / Obsolete

These scripts are kept for reference but should NOT be used:
- `tal_probe_fixed.py` — Uses wrong dataset (IndustRealMultiTaskDataset instead
  of FullMultiModalDataset). Verdict is meaningless. Use `tal_probe_correct.py`.

## Paper SOTA Targets

| Head | Metric | Target |
|------|--------|--------|
| Detection | mAP@0.5 | ≥70% (paper SOTA: 0.641 MViTv2-S) |
| Activity | Top-1 | ≥95% |
| Head Pose | MAE | <5° |
| PSR | F1 | ≥80% |