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
