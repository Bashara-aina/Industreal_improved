# Archive Index

`code/_archive/` contains pre-archive snapshots from earlier cycles. **Not active —
do not run or import from these directories without explicit intent.**

## `code/_archive/archive/` (1.9 GB, dated 2026-05-14)
A full pre-archive snapshot of the codebase from May 14. Predominantly:
- `__pycache__/` directories (regeneratable, ~70% of size)
- Training logs
- Old run outputs
Contents were preserved untouched when moved into `_archive/` on 2026-06-04.

## `code/_archive/popw_main_archived/` (368 KB, dated 2026-05-01)
A smaller pre-archive snapshot from May 1. Source files only — no
`__pycache__/` or logs. Preserved untouched on 2026-06-04.

## Why underscored?
The `_archive/` prefix visually distinguishes the archived directories from the
active project (`industreal_improved/`) and the live analysis packages
(`opus_analysis_package/`, `opus_analysis_package_v3/`). Underscore sorts to top
in `ls`.
