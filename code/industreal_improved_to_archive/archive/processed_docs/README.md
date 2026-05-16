# POPW — Organized Runs Directory

```
runs/
├── pretrain_synthetic/          ← Detection-only pretraining (1 epoch verified)
│   ├── checkpoints/
│   │   └── latest.pth         ← Our 1-epoch test checkpoint (2026-05-07)
│   └── logs/
│       └── pretrain.log
│
└── full_multi_task_tma_tbank/  ← Full multi-task training (active)
    ├── checkpoints/           ← Train.py writes here
    ├── logs/                   ← train.py logs here
    └── eval_outputs/           ← evaluate_all outputs here
```

## Config Paths

All paths are relative to `config.py`:
```python
OUTPUT_ROOT   → runs/full_multi_task_tma_tbank/
CHECKPOINT_DIR → OUTPUT_ROOT / 'checkpoints'
LOG_DIR       → OUTPUT_ROOT / 'logs'
EVAL_SAVE_DIR → OUTPUT_ROOT / 'eval_outputs'
```

## Running Training

```bash
# Full multi-task training (all 5 tasks)
python train.py --epochs 20 --lr 5e-4

# Resume from checkpoint
python train.py --resume runs/full_multi_task_tma_tbank/checkpoints/latest.pth

# Detection-only pretrain (if needed again)
python pretrain_synthetic.py --epochs 20
```

## Archived Old Runs

Old runs from `industreal_improved/` have been moved to:
```
archive/old_runs_20260507/
```
