# POPW Code Directory Reorganization — Design

**Date:** 2026-06-04
**Project:** POPW (Part-Oriented Process Worker) — master's thesis on IndustReal multi-task
assembly recognition
**Working dir:** `/home/newadmin/swarm-bot/project/popw/working/code/`
**Status:** Design approved (user "yes" on 2026-06-04), pending spec review

---

## 1. Problem

The `code/` working directory has accreted three layers of duplication and two layers of
misleading naming over 6+ weeks of debugging iterations:

1. **Triple duplication of the main paper** — `popw_paper_improved.tex` exists as a
   byte-identical 84 KB copy in three places: `code/`, `code/opus_analysis_package/`, and
   `code/opus_analysis_package_v3/paper/`.
2. **LaTeX build noise at the repo root** — `*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`,
   `*.toc`, `*.fls`, `*.fdb_latexmk` are scattered next to the `.tex` files, making the
   root look like a build dir.
3. **Misleading main-folder name** — `code/industreal_improved_to_archive/` is actually
   the **active project** (user confirmed 2026-06-04). The `_to_archive` suffix is a
   legacy artifact from a previous archive cycle.
4. **Real archives mixed with active project** — `code/archive/` (1.9 GB) and
   `code/popw_main_archived/` (368 KB) sit at the same level as the active project and
   one of the Opus handoff packages, with no visual distinction.
5. **Reports scattered** — 5 key `.md` reports (FINAL_REPORT, FIX_REPORT_V2, AUDIT_REPORT,
   MASTER_PROMPT_v3, 01-03_*.md recipes) live 2-3 levels deep. A new contributor cannot
   find them by browsing the root.

## 2. Goal

Make `code/` paper-ready: a reviewer opening the directory should know in under 60 seconds
what the project is, where the code lives, where the paper lives, what is historical, and
where to start. Zero content loss. Zero destruction of regeneratable or historical data
inside the renamed/moved directories.

## 3. Target Structure

```
code/
├── README.md                              ← NEW 1-page orientation
├── .gitignore                             ← NEW: ignore paper/build/ noise
│
├── popw_paper_improved.tex                ← canonical (stays at root, per user)
├── popw_paper.tex                         ← secondary (stays)
├── popwbenchmark.tex                      ← secondary (stays)
├── validation_report.md                   ← stays
├── 18120D-TICKETS.pdf                     ← stays
├── files.zip                              ← stays
│
├── docs/                                  ← NEW: orientation layer
│   ├── README.md                          ← index of every file in code/
│   ├── PAPER_STATUS.md                    ← canonical paper + 2 dupes explained
│   ├── ANALYSIS_INDEX.md                  ← opus v1→v2→v3 chain + popw_main symlink
│   ├── ARCHIVE_INDEX.md                   ← code/_archive/ contents + dates
│   ├── ACTIVE_PROJECT.md                  ← quickstart for industreal_improved/
│   └── reports/                           ← symlinks to scattered .md reports
│       ├── FINAL_REPORT.md                ← → ../../industreal_improved/POPW_FINAL_REPORT.md
│       ├── FIX_REPORT_V2.md               ← → ../../industreal_improved/POPW_FIX_REPORT_V2.md
│       ├── AUDIT_REPORT.md                ← → ../../industreal_improved/AUDIT_REPORT.md
│       ├── MASTER_PROMPT_v3.md            ← → ../../opus_analysis_package_v3/MASTER_PROMPT.md
│       └── RECIPES.md                     ← → ../../files/01_PER_TARGET_RECIPE.md
│
├── paper/                                 ← NEW: LaTeX build isolation
│   ├── popw_paper_improved.tex            ← symlink → ../popw_paper_improved.tex
│   └── build/                             ← moved: .aux .log .bbl .blg .out .toc .fls .fdb_latexmk
│
├── industreal_improved/                   ← RENAMED from industreal_improved_to_archive/
│   └── (existing src/, docs/, debug/, scripts/, results/, opus_*, etc. — untouched)
│
├── opus_analysis_package/                 ← v1, stays as-is
├── opus_analysis_package_v3/              ← v3, stays as-is
│
├── popw_main -> /media/newadmin/master/POPW/popw_main   ← symlink stays
│
├── _archive/                              ← NEW: real archives, underscore prefix
│   ├── README.md                          ← what each subdir is
│   ├── archive/                           ← moved from code/archive/
│   └── popw_main_archived/                ← moved from code/popw_main_archived/
│
└── runs/                                  ← stays
```

## 4. Concrete Moves

| # | Action | Source | Destination | Type |
|---|--------|--------|-------------|------|
| 1 | `git mv` | `code/industreal_improved_to_archive/` | `code/industreal_improved/` | rename |
| 2 | `git mv` | `code/archive/` | `code/_archive/archive/` | move |
| 3 | `git mv` | `code/popw_main_archived/` | `code/_archive/popw_main_archived/` | move |
| 4 | `git mv` (each) | `code/*.aux`, `code/*.log`, `code/*.bbl`, `code/*.blg`, `code/*.out`, `code/*.toc`, `code/*.fls`, `code/*.fdb_latexmk` | `code/paper/build/` | move |
| 5 | `git rm` (per user: redundant copies of paper) | `code/opus_analysis_package/popw_paper_improved.tex` | (delete) | remove |
| 6 | `git rm` (per user: redundant copy of paper) | `code/opus_analysis_package_v3/paper/popw_paper_improved.tex` | (delete) | remove |
| 7 | `ln -s` | `../popw_paper_improved.tex` | `code/paper/popw_paper_improved.tex` | symlink |
| 8 | `Write` | new file | `code/paper/build/.gitkeep` | placeholder (so the dir tracks) |
| 9 | `Write` | new files | `code/docs/README.md`, `code/docs/PAPER_STATUS.md`, `code/docs/ANALYSIS_INDEX.md`, `code/docs/ARCHIVE_INDEX.md`, `code/docs/ACTIVE_PROJECT.md`, `code/docs/reports/*.md` (symlinks), `code/_archive/README.md` | add |
| 10 | `Write` | new file | `code/README.md` | add |
| 11 | `Write` | new file | `code/.gitignore` | add |

**No moves** inside `industreal_improved/`, `opus_analysis_package/`, `opus_analysis_package_v3/`,
or `popw_main` symlink target. **No moves** into the v3 paper/ that would conflict with the
delete in step 6 (since both files are at the same path, the delete happens first).

## 5. Doc File Contents (sketch)

- **`code/README.md`** — 1-page orientation: what is POPW, where the code is, where the
  paper is, where the docs are. Three outbound links: `docs/README.md` (orientation),
  `industreal_improved/` (code), `popw_paper_improved.tex` (paper).
- **`code/docs/README.md`** — index: every file in `code/` with a 1-line description
  grouped by section (Paper, Docs, Code, Analysis, Archive, Reference). Self-contained —
  reads well even if other docs are deleted.
- **`code/docs/PAPER_STATUS.md`** — `popw_paper_improved.tex` at root is canonical.
  `popw_paper.tex` and `popwbenchmark.tex` are previous templates. Opus handoff copies
  removed 2026-06-04 in this reorganization.
- **`code/docs/ANALYSIS_INDEX.md`** — Opus handoff chain: v1 (`opus_analysis_package/`,
  2026-05-26) → v2 inside the active project at `industreal_improved/opus_package_v2/`
  → v3 (`opus_analysis_package_v3/`, 2026-05-31). The v1 and v2 inside the project
  (`industreal_improved/opus_analysis/` and `industreal_improved/opus_analysis_package/`)
  are also preserved. The `popw_main` symlink points to the canonical repo on the
  master disk (`/media/newadmin/master/POPW/popw_main`).
- **`code/docs/ARCHIVE_INDEX.md`** — `_archive/archive/` is a pre-archive snapshot of
  the codebase (mostly `__pycache__/` + logs, 1.9 GB). `_archive/popw_main_archived/`
  is a smaller pre-archive snapshot (368 KB). Both moved into `_archive/` in this
  reorganization, contents preserved.
- **`code/docs/ACTIVE_PROJECT.md`** — quickstart: cd into `industreal_improved/`, the
  source layout (src/{config.py, model.py, data/, training/, evaluation/}), how to run
  the training driver (`src/run_restart_25pct.sh`), where the smoke test is
  (`src/smoke_test_fixes.py`).
- **`code/docs/reports/*.md`** — symlinks. The .md files themselves are not duplicated;
  the symlink is the entry point. (`docs/reports/RECIPES.md` is a single symlink to
  `files/01_PER_TARGET_RECIPE.md`; the other 2 recipe files are listed in
  `RECIPES.md`'s source).
- **`code/_archive/README.md`** — 3-line statement: "Pre-archive snapshots preserved
  for reference. Not active. Contents untouched since 2026-06-04 reorganization. See
  `code/docs/ARCHIVE_INDEX.md`."
- **`code/.gitignore`** — `paper/build/*` except `.gitkeep`. Standard LaTeX noise.

## 6. Git Workflow

Per user: "git add docs + moves only" — only commit the reorganized structure and the
new docs. Do **not** touch git history of the project internals
(`industreal_improved/`, `opus_analysis_package/`, `opus_analysis_package_v3/`).

Single commit at the end with a clear message:

```
chore(code): reorganize code/ for paper-ready delivery

- Rename main project: industreal_improved_to_archive/ → industreal_improved/
- Move real archives into _archive/{archive,popw_main_archived}/
- Isolate LaTeX build noise into paper/build/
- Add docs/ orientation layer (README + 5 indexes + reports/ symlinks)
- Add code/README.md and code/.gitignore
- Remove 2 redundant copies of popw_paper_improved.tex (opus packages)
- Add paper/popw_paper_improved.tex symlink back to canonical at root

No content loss. No changes inside industreal_improved/ or the opus packages.
```

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `git mv` on a directory with many subdirs is slow | Acceptable — single rename, one commit. Time the move, abort if it exceeds 5 min. |
| `git mv` across paths can fail if path is already tracked differently | Verify with `git status` after each move; abort and revert if unexpected. |
| Deleting the 2 paper copies loses git history | Use `git rm` (not `rm`) so the delete is tracked; the blob remains in reflog for 90 days. |
| Symlink `popw_paper_improved.tex` at `paper/` breaks if the file at root is renamed | Document the relationship in `docs/PAPER_STATUS.md`. The symlink is a convenience, not the canonical. |
| A script in `industreal_improved/` references the old path `industreal_improved_to_archive/` | Out of scope for this task per user (lowest risk). Will be flagged in `docs/ACTIVE_PROJECT.md` "Known follow-ups" section. |
| Symlinks in `docs/reports/` break if a target is later moved | Use relative paths (`../../industreal_improved/...`). If a target moves, the symlink breaks loudly with a clear error — easier to detect than a silent copy. |
| Commit is large | Check `git status --short` count before commit. If > 200 files, split into 2 commits: (a) renames + moves, (b) new docs. |

## 8. Out of Scope (Deferred)

- Renaming the `popw_main` symlink target or moving it into the project.
- Removing `__pycache__/`, `*.log.gz`, or any other regeneratable noise inside the
  archived directories (user explicitly chose "keep all contents").
- Updating absolute paths inside scripts (`run_*.sh` files referencing
  `/media/newadmin/master/POPW/...`) — that's a project-internal change.
- Consolidating Opus v1 + v3 — user wants both kept.
- Fixing the smoke test typo (`ema` → `ema_after` on line 619) — that's a code bug,
  not an organization issue.

## 9. Acceptance

- [ ] `git status` shows only the planned renames, moves, deletes, and new docs.
- [ ] `code/README.md` opens with 3 outbound links and renders as a project landing page.
- [ ] `code/docs/reports/*.md` symlinks all resolve when followed with `readlink -f`.
- [ ] `paper/build/` contains the moved LaTeX noise; no `.aux` / `.log` / `.bbl` at root.
- [ ] `industreal_improved/` is the canonical name; old name is gone.
- [ ] `_archive/` contains the 2 real archives; `code/archive/` and `code/popw_main_archived/`
      are gone.
- [ ] Single commit (or 2 if file count > 200) on the working branch.
- [ ] User reviews the diff via `git show --stat HEAD` and approves.

---

*End of design. Awaiting user review of the written spec before any moves.*
