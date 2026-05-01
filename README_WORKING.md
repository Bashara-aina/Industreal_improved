# WORKING Hub

This folder is the canonical organized hub for active training and data usage.

## Structure

- `code/popw_main` -> main unified training codebase
- `data/datasets` -> prepared dataset links for modes (`manual_only`, `manual_pseudo`)
- `data/IKEA_RAW` -> raw annotations + pose predictions + COCO roots
- `data/IKEA_dataset` -> source image tree (dev3 and related layout)
- `external/github` -> IKEA toolbox/split files dependency
- `external/ikea_workernet_FULL` -> action lookup dependency
- `artifacts/model` -> model checkpoints store
- `artifacts/runs` -> global runs outputs
- `artifacts/popw_main_runs` -> unified popw_main run outputs
- `archive/ARCHIVE` -> archived legacy variants
- `docs/MASTER_PLAN_COMPLIANCE_REPORT.md` -> implementation checklist report

## Run from WORKING

```bash
cd /media/newadmin/master/POPW/working/code/popw_main
python train.py --preset improved4
```

## Validate from WORKING

```bash
cd /media/newadmin/master/POPW/working/code/popw_main
python scripts/validate_multi_camera.py
python test_dataloader.py --dataset manual_only --detection all_cameras --max-batches 1
```

## Notes

- This setup is non-destructive: original folders stay in place.
- `working` uses symlinks so updates remain synchronized automatically.
