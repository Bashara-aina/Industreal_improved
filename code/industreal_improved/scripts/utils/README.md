# Utility Scripts

General-purpose tools and helpers.

## Scripts

- `export_onnx.py` — Export model to ONNX format for deployment
- `profile_dataloader.py` — Profile DataLoader throughput
- `mediapipe_pose_baseline.py` — MediaPipe pose baseline (reference comparison)
- `training_monitor.py` — Detect anomalies in running training
- `mvp_smoke_suite.py` — MVP smoke suite (diagnose 0.0/0.008 metrics)
- `minimal_smoke_test.py` — Minimal smoke test
- `smoke_test.py` — Full smoke test
- `smoke_test_4heads.py` — 4-head smoke test
- `run_q43_canonical_pos.py` — Q43 canonical-order POS baseline

## Shell Scripts (`.sh`)

Training management and automation:
- `kill_training.sh` — Kill running training processes
- `monitor_training.sh` — Monitor training progress (tail logs)
- `freeze_checkpoint.sh` — Freeze a checkpoint
- `launch_full_training_pipeline.sh` — Launch full pipeline
- `launch_r25_fix.sh` — Launch R25 fix training
- `launch_st_baselines.sh` — Launch single-task baselines
- `launch_uncapped_kendall.sh` — Launch uncapped Kendall training
- `restart_r3_training.sh` — Restart R3 training
- `restart_rf2_training.sh` — Restart RF2 training
- `full_pipeline_v1.sh` — Full v1 pipeline
- `reproduce.sh` — Reproducibility script
- `download_yolov8m_industreal.sh` — Download YOLOv8 weights

## Other Files

- `training_pids.txt` — Last training PIDs (auto-generated)