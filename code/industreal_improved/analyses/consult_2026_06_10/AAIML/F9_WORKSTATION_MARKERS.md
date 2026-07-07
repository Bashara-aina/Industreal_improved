# F9 — Workstation-Only Markers Audit

**Date:** 2026-07-07
**Source:** File-157 audit — claims unverifiable from remote GitHub repo
**Performed by:** Agent 101 (F-9 Workstation-Only Markers Specialist)

---

## Summary

This document catalogs all claims in files 150-156 that depend on workstation-local artifacts and cannot be verified from the remote GitHub repository. Each such claim has been annotated with an `UNVERIFIABLE-REMOTELY` marker in the source files.

---

## Dependent Artifacts

| Artifact | Location | Verifiable from GitHub? |
|---|---|---|
| `best.pth` (738MB, SHA256) | `src/runs/rf_stages/checkpoints/best.pth` | NO — checkpoint not in git (738MB, gitignored) |
| `crash_recovery.pth` (738MB) | `src/runs/rf_stages/checkpoints/crash_recovery.pth` | NO — checkpoint not in git |
| `/tmp/train_psr_repair_v3.log` | Workstation `/tmp/` | NO — temp file, not persisted |
| `/tmp/train_singletask_det.log` | Workstation `/tmp/` | NO — temp file, not persisted |
| `/tmp/temporal_probe_cpu.log` | Workstation `/tmp/` | NO — temp file, not persisted |
| V3 training process state | Running on 5060 Ti | NO — process state ephemeral |
| Single-task detection training state | Running on 5060 Ti | NO — process state ephemeral |
| Post_gelu activation values | Parsed from `/tmp/*.log` | NO — derived from ephemeral logs |
| Epoch counts (24+, 43+, etc.) | Parsed from `/tmp/*.log` | NO — live process state |
| Activation +4608 value | `/tmp/train_psr_repair_v3.log` | NO — from ephemeral log |

---

## Files 150-156: Marker Count Per File

| File | Markers Added |
|---|---|
| `150_MASTER_SYNTHESIS.md` | See inline annotations |
| `150_SOTA_STATUS_V5.md` | See inline annotations |
| `151_PER_HEAD_DEEP_ANALYSIS.md` | See inline annotations |
| `152_IMPLEMENTATION_BUG_CATALOG.md` | See inline annotations |
| `153_MULTI_TASK_DEBATE.md` | See inline annotations |
| `154_SOTA_COMPARISON.md` | See inline annotations |
| `155_FINAL_PAPER_NARRATIVE.md` | See inline annotations |
| `156_100_DEEP_QUESTIONS.md` | See inline annotations |

---

## Detailed Claim Inventory

### 150_MASTER_SYNTHESIS.md — 14 workstation-dependent claims

1. L40: best.pth SHA256 `59cb88ec...` — checkpoint not in git
2. L60: `/tmp/train_psr_repair_v3.log` — post_gelu +4608, epochs 24+
3. L61: `/tmp/train_singletask_det.log` — epoch 43+, ~3.4 days remaining
4. L121: V3 running NOW — post_gelu +4608, epochs 24+
5. L123: Single-task detection running — epochs 43+, ~3.4 days
6. L139: Single-task detection epoch 43+, ~3.4 days
7. L163: Single-task detection ~3.4 days
8. L272: Single-task detection epoch 43+, ~3.4 days
9. L296: V3 ~2 days from epoch 24+, single-task ~3.4 days
10. L317: V3 running NOW, epochs 24+
11. L321: best.pth SHA256 verification
12. L323: Single-task detection running, epochs 43+
13. L457: `/tmp/train_psr_repair_v3.log`
14. L458: `/tmp/train_singletask_det.log`

### 150_SOTA_STATUS_V5.md — 2 workstation-dependent claims

1. L54: PSR head repair training epoch 24+, activations alive
2. L55: Single-task ConvNeXt detection epoch 24+

### 151_PER_HEAD_DEEP_ANALYSIS.md — 5 workstation-dependent claims

1. L75: Single-task training epoch 43+, ~3.4 days
2. L100: Post_gelu activations -130 to +4608
3. L108: `/tmp/train_psr_repair_v3.log`
4. L113: V3 training epoch 25+, post_gelu +4608
5. L168: Single-task detection 3-4 days

### 152_IMPLEMENTATION_BUG_CATALOG.md — 3 workstation-dependent claims

1. L37: post_gelu activations +4608 (was -1.0 to -1.4 dead)
2. L43: `/tmp/train_psr_repair_v3.log`
3. L144: Activations -130 -> +4608

### 153_MULTI_TASK_DEBATE.md — 2 workstation-dependent claims

1. L73: Single-task detection ~3.4 days remaining
2. L143: Get single-task detection results (3-4 days)

### 154_SOTA_COMPARISON.md — 1 workstation-dependent claim

1. L221: epoch_18 best.pth reference

### 155_FINAL_PAPER_NARRATIVE.md — 4 workstation-dependent claims

1. L42: post_gelu -130 to +4608 on sequence frames
2. L95: post-GELU activations -1.0 to -1.4 to +4608
3. L97: V3 PSR repair training in flight
4. L176: best.pth SHA256 `59cb88ec...`

### 156_100_DEEP_QUESTIONS.md — 4 workstation-dependent claims

1. L311: Post_gelu -1.0 to -1.4 to +4608
2. L312: Source `/tmp/train_psr_repair_v3.log`
3. L531: Activations -1.0 to +4608
4. L445: V3 PSR in flight

---

## Total: 35 workstation-dependent claims across 8 files

All marked with `(UNVERIFIABLE-REMOTELY)` annotations in source files.
