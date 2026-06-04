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
