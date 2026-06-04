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
