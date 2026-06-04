# code/ — File Index

Every file in `code/` with a one-line description, grouped by section. Read this first
if you're new to the project.

## Paper (root)
- `popw_paper_improved.tex` — Canonical master's thesis (84 KB). **The paper.**
- `popw_paper.tex` — Earlier template (superseded).
- `popwbenchmark.tex` — Benchmark template (superseded).
- `validation_report.md` — 15 baseline claims verified.
- `18120D-TICKETS.pdf` — Reference ticket data.
- `files.zip` — Bundled reference files.
- `paper/popw_paper_improved.tex` — Symlink to canonical paper at root.

## Docs (this directory)
- `docs/README.md` — This file.
- `docs/PAPER_STATUS.md` — Why 3 paper copies existed, which is canonical.
- `docs/ANALYSIS_INDEX.md` — Opus handoff chain v1→v2→v3.
- `docs/ARCHIVE_INDEX.md` — What's in `_archive/`.
- `docs/ACTIVE_PROJECT.md` — Quickstart for the active code project.
- `docs/reports/*.md` — Symlinks to key reports (FINAL, FIX, AUDIT, MASTER_PROMPT, RECIPES).

## Code (active project)
- `industreal_improved/` — **Active project** (renamed from `industreal_improved_to_archive/`
  on 2026-06-04). See `docs/ACTIVE_PROJECT.md` to start.

## Analysis (Opus handoffs)
- `opus_analysis_package/` — v1 (2026-05-26).
- `opus_analysis_package_v3/` — v3 (2026-05-31). See `docs/ANALYSIS_INDEX.md`.

## Reference
- `popw_main` — Symlink to canonical repo on master disk
  (`/media/newadmin/master/POPW/popw_main`).
- `runs/` — Training run outputs (checkpoints, logs, metrics).
- `_archive/` — Pre-archive snapshots, preserved. See `docs/ARCHIVE_INDEX.md`.

## Orientation
New here? Read in this order: this file → `docs/ACTIVE_PROJECT.md` → `popw_paper_improved.tex`.
