# POPW Code Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `/home/newadmin/swarm-bot/project/popw/working/code/` to be paper-ready — a reviewer opening the directory should know in under 60 seconds what the project is, where the code is, where the paper is, what is historical, and where to start. Zero content loss.

**Architecture:** This is a structural reorganization, not feature work. We use `git mv` for tracked renames/moves (preserves history), `git rm` for tracked deletes, and `Write` for new files. A single commit at the end encapsulates the move.

**Tech Stack:** Git worktree at `/home/newadmin/swarm-bot/project/popw/working/code/` (worktree of `/media/newadmin/master/POPW/working`). Spec already committed as `df0e645` on `main`.

---

## Important Constraints

- **All `git` commands run from** `/home/newadmin/swarm-bot/project/popw/working/code/`
- **The 200+ unrelated modified files in `.claude-flow/` and `.claude/` must NOT be committed.** They are out of scope. We commit only the moves + new docs.
- **Verification after every step** with `git status --short | grep -v -E '^(\?\? )?(M|\s) \.(claude|claude-flow)/' | head -20` to ensure no scope creep.
- **If `git status` shows unexpected renames/deletes, STOP and investigate before proceeding.**
- **At Task 13, before commit, run the full verification block to confirm only planned files are staged.**

---

## Task 1: Rename main project directory

**Files:**
- Rename: `industreal_improved_to_archive/` → `industreal_improved/`

- [ ] **Step 1: Verify source exists and is tracked**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
ls -la industreal_improved_to_archive/ | head -5
git ls-files industreal_improved_to_archive/ | head -3
```

Expected: directory listing with `src/`, `docs/`, `scripts/`, `runs/` etc.; `git ls-files` returns tracked files like `src/run_restart_25pct.sh`.

- [ ] **Step 2: Verify destination does NOT exist**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
test ! -e industreal_improved && echo "OK: no conflict" || echo "FAIL: already exists"
```

Expected: `OK: no conflict`.

- [ ] **Step 3: git mv rename**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git mv industreal_improved_to_archive industreal_improved
```

- [ ] **Step 4: Verify rename**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
test -d industreal_improved && echo "OK: renamed" || echo "FAIL"
git status --short | grep -E "industreal_improved(_to_archive)?/" | head -5
```

Expected: `OK: renamed`; `git status` shows `R  industreal_improved_to_archive/ -> industreal_improved/`.

---

## Task 2: Move real archive into _archive/

**Files:**
- Move: `archive/` → `_archive/archive/`

- [ ] **Step 1: Create _archive/ directory**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
mkdir -p _archive
```

- [ ] **Step 2: Verify source is tracked**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git ls-files archive/ | head -3
du -sh archive/
```

Expected: at least one tracked file; size ~1.9 GB.

- [ ] **Step 3: git mv into _archive/**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git mv archive/ _archive/archive
```

- [ ] **Step 4: Verify move**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
test -d _archive/archive && test ! -e archive && echo "OK"
```

Expected: `OK`.

---

## Task 3: Move popw_main_archived into _archive/

**Files:**
- Move: `popw_main_archived/` → `_archive/popw_main_archived/`

- [ ] **Step 1: Verify source**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
test -d popw_main_archived && echo "OK" || echo "FAIL: missing"
git ls-files popw_main_archived/ | head -3
du -sh popw_main_archived/
```

Expected: `OK`; size ~368 KB.

- [ ] **Step 2: git mv**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git mv popw_main_archived/ _archive/popw_main_archived
```

- [ ] **Step 3: Verify**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
test -d _archive/popw_main_archived && test ! -e popw_main_archived && echo "OK"
```

Expected: `OK`.

---

## Task 4: Create _archive/README.md

**Files:**
- Create: `_archive/README.md`

- [ ] **Step 1: Write the README**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/_archive/README.md <<'EOF'
# _archive/ — Pre-archive Snapshots

Preserved snapshots of the codebase from earlier archive cycles. **Not active.**

## Contents

- `archive/` — Large snapshot (1.9 GB) from 2026-05-14. Mostly `__pycache__/` and
  training logs. Contents untouched.
- `popw_main_archived/` — Smaller snapshot (368 KB) from 2026-05-01. Contents untouched.

## Origin

Both directories were moved into `_archive/` on 2026-06-04 during the `code/`
reorganization (commit `df0e645` design spec → implementation commit).

## See also

- `code/docs/ARCHIVE_INDEX.md` — index of what's in each archive subdirectory.
EOF
```

- [ ] **Step 2: Verify file created**

```bash
ls -la /home/newadmin/swarm-bot/project/popw/working/code/_archive/README.md
```

Expected: file exists with content.

---

## Task 5: Create paper/ directory and build/ subdirectory

**Files:**
- Create: `paper/` directory
- Create: `paper/build/` directory

- [ ] **Step 1: Create directories**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
mkdir -p paper/build
```

- [ ] **Step 2: Verify**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
test -d paper && test -d paper/build && echo "OK"
```

Expected: `OK`.

---

## Task 6: Move LaTeX build noise into paper/build/

**Files:**
- Move all `*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`, `*.toc` from `code/` root to `code/paper/build/`

- [ ] **Step 1: List current LaTeX noise files**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
ls *.aux *.log *.bbl *.blg *.out *.toc 2>/dev/null
```

Expected: list of files like `popw_paper_improved.aux`, `popw_paper_improved.log`, etc. (no output is fine if user cleaned; skip step 2-3 if empty).

- [ ] **Step 2: git mv each file into paper/build/**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
for f in *.aux *.log *.bbl *.blg *.out *.toc; do
  if [ -f "$f" ]; then
    git mv "$f" paper/build/
  fi
done
```

- [ ] **Step 3: Verify no LaTeX noise at root**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
ls *.aux *.log *.bbl *.blg *.out *.toc 2>/dev/null | wc -l
ls paper/build/ | head -10
```

Expected: `0` (no files at root); `paper/build/` contains the moved files.

---

## Task 7: Create paper/build/.gitkeep

**Files:**
- Create: `paper/build/.gitkeep`

- [ ] **Step 1: Write .gitkeep**

```bash
touch /home/newadmin/swarm-bot/project/popw/working/code/paper/build/.gitkeep
```

- [ ] **Step 2: Verify**

```bash
ls -la /home/newadmin/swarm-bot/project/popw/working/code/paper/build/.gitkeep
```

Expected: file exists.

---

## Task 8: Add paper/ symlink to canonical paper

**Files:**
- Create symlink: `paper/popw_paper_improved.tex` → `../popw_paper_improved.tex`

- [ ] **Step 1: Create symlink**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
cd paper && ln -s ../popw_paper_improved.tex popw_paper_improved.tex && cd ..
```

- [ ] **Step 2: Verify symlink**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
ls -la paper/popw_paper_improved.tex
readlink -f paper/popw_paper_improved.tex
```

Expected: symlink shows arrow `paper/popw_paper_improved.tex -> ../popw_paper_improved.tex`; `readlink -f` resolves to `/home/newadmin/swarm-bot/project/popw/working/code/popw_paper_improved.tex`.

---

## Task 9: Remove redundant paper copies

**Files:**
- Delete: `opus_analysis_package/popw_paper_improved.tex`
- Delete: `opus_analysis_package_v3/paper/popw_paper_improved.tex`

- [ ] **Step 1: Verify the two files are byte-identical to canonical**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
md5sum popw_paper_improved.tex \
       opus_analysis_package/popw_paper_improved.tex \
       opus_analysis_package_v3/paper/popw_paper_improved.tex
```

Expected: all three MD5 hashes match. **If they don't match, STOP and investigate — do not delete a non-identical file.**

- [ ] **Step 2: git rm both copies**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git rm opus_analysis_package/popw_paper_improved.tex
git rm opus_analysis_package_v3/paper/popw_paper_improved.tex
```

- [ ] **Step 3: Verify only canonical paper remains**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
find . -name popw_paper_improved.tex -not -path "./.git/*"
```

Expected: exactly 2 results: `./popw_paper_improved.tex` (canonical) and `./paper/popw_paper_improved.tex` (symlink).

---

## Task 10: Create docs/ layer — orientation files

**Files:**
- Create: `docs/README.md`
- Create: `docs/PAPER_STATUS.md`
- Create: `docs/ANALYSIS_INDEX.md`
- Create: `docs/ARCHIVE_INDEX.md`
- Create: `docs/ACTIVE_PROJECT.md`

- [ ] **Step 1: Create docs/ directory**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
mkdir -p docs/reports
```

- [ ] **Step 2: Write docs/README.md (index of every file in code/)**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/docs/README.md <<'EOF'
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
EOF
```

- [ ] **Step 3: Write docs/PAPER_STATUS.md**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/docs/PAPER_STATUS.md <<'EOF'
# Paper Status

**Canonical paper:** `code/popw_paper_improved.tex` (84,148 bytes as of 2026-06-04).

## History
Until 2026-06-04, the paper existed as a byte-identical copy in three places:
1. `code/popw_paper_improved.tex` — canonical, at the repo root for reviewer convenience
2. `code/opus_analysis_package/popw_paper_improved.tex` — sent to Opus v1 (2026-05-26)
3. `code/opus_analysis_package_v3/paper/popw_paper_improved.tex` — sent to Opus v3 (2026-05-31)

On 2026-06-04, copies #2 and #3 were deleted (verified byte-identical via md5sum before
deletion). The canonical at root remains. A symlink at `code/paper/popw_paper_improved.tex`
points back to the canonical for convenience.

## Earlier templates (kept for reference)
- `code/popw_paper.tex` — initial template (superseded by `_improved` variant)
- `code/popwbenchmark.tex` — benchmark template (used for baseline reporting)

## Modifications
All paper edits go to the canonical at `code/popw_paper_improved.tex`. The
`code/paper/` symlink follows automatically — there is only one paper file.
EOF
```

- [ ] **Step 4: Write docs/ANALYSIS_INDEX.md**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/docs/ANALYSIS_INDEX.md <<'EOF'
# Opus Handoff Analysis Chain

POPW has been analyzed by Opus three times during the project. Each handoff package
is preserved.

## v1 — `code/opus_analysis_package/` (2026-05-26)
- Original Opus handoff. Sent first to request deep code review.
- Contains a redundant copy of `popw_paper_improved.tex` (deleted 2026-06-04).

## v2 — `code/industreal_improved/opus_package_v2/` (2026-05-28)
- Internal to the active project. Includes bug-fix proposals from the v1 review.
- Note: there are *two* additional internal v2 handoffs inside the active project:
  - `code/industreal_improved/opus_analysis/` (older, single file)
  - `code/industreal_improved/opus_analysis_package/` (newer, full package)

## v3 — `code/opus_analysis_package_v3/` (2026-05-31)
- Most recent. Generated after Stage 1/2/3 wiring and the 15-bug audit.
- `MASTER_PROMPT.md` at the root is the primary context for any future Opus call.
- Contains a redundant copy of `popw_paper_improved.tex` (deleted 2026-06-04).

## Canonical repo
The `code/popw_main` symlink points to `/media/newadmin/master/POPW/popw_main`,
the canonical repo on the master disk. `code/` itself is a worktree of
`/media/newadmin/master/POPW/working`.
EOF
```

- [ ] **Step 5: Write docs/ARCHIVE_INDEX.md**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/docs/ARCHIVE_INDEX.md <<'EOF'
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
EOF
```

- [ ] **Step 6: Write docs/ACTIVE_PROJECT.md**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/docs/ACTIVE_PROJECT.md <<'EOF'
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
EOF
```

- [ ] **Step 7: Verify all 5 files exist**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
ls docs/README.md docs/PAPER_STATUS.md docs/ANALYSIS_INDEX.md docs/ARCHIVE_INDEX.md docs/ACTIVE_PROJECT.md
```

Expected: 5 lines, all paths printed.

---

## Task 11: Create docs/reports/ symlinks

**Files:**
- Symlink: `docs/reports/FINAL_REPORT.md` → `../../industreal_improved/POPW_FINAL_REPORT.md`
- Symlink: `docs/reports/FIX_REPORT_V2.md` → `../../industreal_improved/POPW_FIX_REPORT_V2.md`
- Symlink: `docs/reports/AUDIT_REPORT.md` → `../../industreal_improved/AUDIT_REPORT.md`
- Symlink: `docs/reports/MASTER_PROMPT_v3.md` → `../../opus_analysis_package_v3/MASTER_PROMPT.md`
- Symlink: `docs/reports/RECIPES.md` → `../../files/01_PER_TARGET_RECIPE.md`

- [ ] **Step 1: Verify each target file exists before symlinking**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
for tgt in \
  industreal_improved/POPW_FINAL_REPORT.md \
  industreal_improved/POPW_FIX_REPORT_V2.md \
  industreal_improved/AUDIT_REPORT.md \
  opus_analysis_package_v3/MASTER_PROMPT.md \
  files/01_PER_TARGET_RECIPE.md ; do
  if [ ! -f "$tgt" ]; then echo "MISSING: $tgt"; else echo "OK: $tgt"; fi
done
```

Expected: 5 `OK:` lines. **If any MISSING, STOP — adjust the symlink or skip that link.**

- [ ] **Step 2: Create symlinks**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/docs/reports
ln -s ../../industreal_improved/POPW_FINAL_REPORT.md  FINAL_REPORT.md
ln -s ../../industreal_improved/POPW_FIX_REPORT_V2.md  FIX_REPORT_V2.md
ln -s ../../industreal_improved/AUDIT_REPORT.md        AUDIT_REPORT.md
ln -s ../../opus_analysis_package_v3/MASTER_PROMPT.md MASTER_PROMPT_v3.md
ln -s ../../files/01_PER_TARGET_RECIPE.md             RECIPES.md
```

- [ ] **Step 3: Verify all symlinks resolve**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
for f in docs/reports/*.md; do
  resolved=$(readlink -f "$f")
  test -f "$resolved" && echo "OK: $f -> $resolved" || echo "BROKEN: $f -> $resolved"
done
```

Expected: 5 `OK:` lines.

---

## Task 12: Create code/README.md and code/.gitignore

**Files:**
- Create: `README.md` (at code/ root)
- Create: `.gitignore` (at code/ root)

- [ ] **Step 1: Write code/README.md**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/README.md <<'EOF'
# POPW — Part-Oriented Process Worker

**Master's thesis project** on IndustReal multi-task assembly recognition.
This `code/` directory is the working copy; the canonical repo lives at
`/media/newadmin/master/POPW/popw_main` (also accessible as the `popw_main` symlink here).

## Start here

- **New to the project?** Read [`docs/README.md`](docs/README.md) for a file-by-file
  index of this directory.
- **Want to run something?** Read [`docs/ACTIVE_PROJECT.md`](docs/ACTIVE_PROJECT.md) for
  the active code project's quickstart (smoke test, restart training).
- **Reading the paper?** It's at [`popw_paper_improved.tex`](popw_paper_improved.tex)
  (also symlinked at `paper/popw_paper_improved.tex`).

## Tasks

POPW trains a single multi-task model on 4 IndustReal tasks:

1. **ASD** — Assembly State Detection (24 classes, bounding box)
2. **Activity Recognition** — 75 classes (NA + IDs 1-74)
3. **Head Pose Estimation** — 9-DoF regression (forward, position, up)
4. **PSR** — Procedure State Recognition (temporal, 36 steps, 11 components)

Architecture: ConvNeXt-Tiny backbone + FPN neck + 4 task-specific heads, with TMA
temporal cell, Feature Bank (T=16), Hand-FiLM, and HeadPoseFiLM conditioning.

## Layout (one-line)

| Path | What |
|------|------|
| `popw_paper_improved.tex` | The thesis paper. |
| `docs/` | Orientation layer (this README, indexes, report symlinks). |
| `paper/` | LaTeX build isolation (symlink to canonical paper + build noise dir). |
| `industreal_improved/` | **Active project** (code, scripts, internal docs). |
| `opus_analysis_package/`, `opus_analysis_package_v3/` | Opus handoff packages v1, v3. |
| `popw_main` | Symlink to canonical repo on master disk. |
| `runs/` | Training run outputs. |
| `_archive/` | Pre-archive snapshots (preserved, not active). |

See [`docs/README.md`](docs/README.md) for full file inventory.
EOF
```

- [ ] **Step 2: Write code/.gitignore**

```bash
cat > /home/newadmin/swarm-bot/project/popw/working/code/.gitignore <<'EOF'
# LaTeX build noise — kept in paper/build/, not tracked at root
*.aux
*.log
*.bbl
*.blg
*.out
*.toc
*.fls
*.fdb_latexmk
*.synctex.gz

# Allow paper/build/ directory to be tracked (for .gitkeep)
!paper/build/.gitkeep
EOF
```

- [ ] **Step 3: Verify both files exist**

```bash
ls -la /home/newadmin/swarm-bot/project/popw/working/code/README.md \
       /home/newadmin/swarm-bot/project/popw/working/code/.gitignore
```

Expected: both files exist.

---

## Task 13: Pre-commit verification

- [ ] **Step 1: Run the full scope check (this is the gate before commit)**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
echo "=== Staged + unstaged (planned scope only) ==="
git status --short | grep -v -E '^\s*M\s+(\.claude|\.claude-flow)/' | head -50
echo "=== Untracked count ==="
git status --short | grep '^??' | wc -l
echo "=== Symlink sanity ==="
for f in docs/reports/*.md paper/popw_paper_improved.tex; do
  resolved=$(readlink -f "$f" 2>/dev/null)
  test -e "$resolved" && echo "OK: $f" || echo "BROKEN: $f"
done
echo "=== Root LaTeX noise ==="
ls *.aux *.log *.bbl *.blg *.out *.toc 2>/dev/null | wc -l
echo "=== Active project rename ==="
test -d industreal_improved && test ! -e industreal_improved_to_archive && echo "OK"
echo "=== Archive relocation ==="
test -d _archive/archive && test -d _archive/popw_main_archived && test ! -e archive && test ! -e popw_main_archived && echo "OK"
echo "=== Paper copies ==="
find . -name popw_paper_improved.tex -not -path "./.git/*"
```

Expected output: all `OK` lines; 0 LaTeX noise at root; 2 results from `find` (canonical + symlink).

- [ ] **Step 2: Stage ONLY the planned files (do not use `git add -A` or `git add .`)**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
# Renames (git already tracks these from git mv)
git add -u industreal_improved_to_archive industreal_improved archive _archive popw_main_archived \
        opus_analysis_package/popw_paper_improved.tex \
        opus_analysis_package_v3/paper/popw_paper_improved.tex \
        paper/build \
        popw_paper_improved.aux popw_paper_improved.log popw_paper_improved.bbl \
        popw_paper_improved.blg popw_paper_improved.out popw_paper_improved.toc \
        popw_paper.aux popw_paper.log popw_paper.bbl popw_paper.blg \
        popw_paper.out popw_paper.toc \
        popw_benchmark.aux popw_benchmark.log popw_benchmark.out
# New untracked files
git add README.md .gitignore \
        docs/README.md docs/PAPER_STATUS.md docs/ANALYSIS_INDEX.md \
        docs/ARCHIVE_INDEX.md docs/ACTIVE_PROJECT.md \
        docs/reports/FINAL_REPORT.md docs/reports/FIX_REPORT_V2.md \
        docs/reports/AUDIT_REPORT.md docs/reports/MASTER_PROMPT_v3.md \
        docs/reports/RECIPES.md \
        _archive/README.md \
        paper/build/.gitkeep
```

- [ ] **Step 3: Verify staged set matches the spec exactly**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git status --short | grep -E '^[MAD]\s' | head -60
echo "=== Total staged changes ==="
git status --short | grep -E '^[MAD]\s' | wc -l
```

Expected: ~30-40 staged items, all matching the plan. **If `.claude-flow/` or `.claude/`
files appear in the staged set, STOP and `git restore --staged <path>` to unstage them,
then investigate before proceeding.**

- [ ] **Step 4: Commit with the prepared message**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git commit -m "$(cat <<'EOF'
chore(code): reorganize code/ for paper-ready delivery

- Rename main project: industreal_improved_to_archive/ -> industreal_improved/
- Move real archives into _archive/{archive,popw_main_archived}/
- Isolate LaTeX build noise into paper/build/
- Add docs/ orientation layer (README + 5 indexes + reports/ symlinks)
- Add code/README.md and code/.gitignore
- Remove 2 redundant copies of popw_paper_improved.tex (opus packages)
- Add paper/popw_paper_improved.tex symlink back to canonical at root

No content loss. No changes inside industreal_improved/ or the opus packages.
EOF
)"
```

- [ ] **Step 5: Verify commit landed correctly**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git log --oneline -2
git show --stat HEAD | head -50
```

Expected: new commit on top of `df0e645`; stat shows the expected renames/moves/adds/deletes.

- [ ] **Step 6: Confirm working tree is clean except for the out-of-scope files**

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code
git status --short | grep -v -E '^\s*M\s+(\.claude|\.claude-flow)/' | head -20
echo "=== Done. Out-of-scope .claude/ and .claude-flow/ modifications are untouched. ==="
```

Expected: empty output (or only items we intended to leave).

---

## Acceptance criteria

- [ ] `git status` shows only the planned renames, moves, deletes, and new docs.
- [ ] `code/README.md` opens with 3 outbound links and renders as a project landing page.
- [ ] `code/docs/reports/*.md` symlinks all resolve when followed with `readlink -f`.
- [ ] `paper/build/` contains the moved LaTeX noise; no `.aux` / `.log` / `.bbl` at root.
- [ ] `industreal_improved/` is the canonical name; old name is gone.
- [ ] `_archive/` contains the 2 real archives; `code/archive/` and `code/popw_main_archived/`
      are gone.
- [ ] Single commit (or 2 if file count > 200) on the working branch.
- [ ] User reviews the diff via `git show --stat HEAD` and approves.
