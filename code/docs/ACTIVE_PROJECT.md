# Active Project — Quickstart

**Active project:** `code/industreal_improved/` (renamed from
`industreal_improved_to_archive/` on 2026-06-04).

## What's inside

```
industreal_improved/
├── src/
│   ├── config.py            ← all hyperparameters
│   ├── model.py             ← POPWMultiTaskModel architecture
│   ├── data/                ← dataset implementations
│   ├── training/
│   │   ├── train.py         ← main training loop
│   │   └── losses.py        ← all loss functions + Kendall
│   ├── evaluation/          ← metric computation
│   ├── run_restart_25pct.sh ← restart driver (resumable 25% subset run)
│   ├── smoke_test_fixes.py  ← 16/16 smoke test for V2 fixes
│   └── validate_checkpoint.py
├── docs/                    ← project-internal docs (POPW_FINAL_REPORT, etc.)
├── debug/                   ← debugging scripts
├── scripts/                 ← utility scripts
├── results/                 ← evaluation outputs
├── opus_*/                  ← earlier internal Opus handoffs
├── runs/                    ← training run checkpoints + logs
└── [reference files: README.md, requirements.txt, etc.]
```

## How to run a smoke test

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved
python3 src/smoke_test_fixes.py
```

Expected: `16/16 checks passed` (live).

## How to restart a 25% subset training

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved
bash src/run_restart_25pct.sh
```

Resumes from `src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth`.

## Key reports inside the project

- `POPW_FINAL_REPORT.md` — aggregated verifier results + risk assessment
- `POPW_FIX_REPORT_V2.md` — 16/16 smoke check output
- `AUDIT_REPORT.md` — 15-bug audit results

(These are also accessible via symlinks from `code/docs/reports/`.)

## Known follow-ups (out of scope for the 2026-06-04 reorganization)
- The directory was previously named `industreal_improved_to_archive/`. A small number
  of internal scripts may still reference the old path. If you hit a `No such file`
  error, check whether the script hardcodes the old name.
- The smoke test has a 1-character typo at line 619 (`ema` → `ema_after`).
  Does not affect production code; see `POPW_FIX_REPORT_V2.md` for the fix.
