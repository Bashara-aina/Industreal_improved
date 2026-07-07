# Checkpoint Manifest — SHA256 Hashes

**Purpose:** Per Opus-165 §4.5, the 738 MB `best.pth` cannot be committed to git directly,
but its hash can — so reviewers can verify they have the same model weights that
produced the headline numbers in the paper.

**Method:** `sha256sum` computed 2026-07-07 on the workstation.
Reproduce with: `sha256sum -c CHECKPOINT_MANIFEST.md` after downloading the
checkpoint into the same path.

---

## Epoch 18 — best.pth (full multi-task, headline numbers source)

This is the multi-task checkpoint that produced:
- D3 detection: mAP50 = 0.00009 (broken — 9 fixes applied; see paper)
- D1R detection: mAP50 = 0.995 (single-task YOLOv8m on the same split; derived via `d1_yolov8m_v3/metrics.json`)
- Head pose: fwd MAE 9.14°, up MAE 7.78° (from `full_eval_ep18_v2/metrics.json`)
- PSR: F1 = 0.7018 per-component optimal, F1 = 0.6788 global 0.10 (from `psr_optimal_thr_38k/optimal_thresholds.json`)
- Activity: top-1 = 0.0233 valid-frame / 0.0191 all-frame (class collapse; 0.3810 with frozen MViTv2-S probe)

| File | Size (bytes) | SHA256 |
|---|---|---|
| `src/runs/rf_stages/checkpoints/best.pth` | 738,040,165 | `59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8` |

**Last modified:** 2026-07-06 00:26 (workstation)

**Origin:** Trained on RTX 3060 + RTX 5060 Ti. ConvNeXt-T backbone (linear probe mode),
3-head multi-task (det + head_pose + psr), pre-LeakyReLU repair. The activity
head in this checkpoint is class-collapsed (predicts majority class).

**To verify:** `sha256sum src/runs/rf_stages/checkpoints/best.pth` and compare
to the hash above. If they match, you have the same weights that produced
the headline numbers in `SOTA_STATUS.md` and the paper narrative.

---

## V3 PSR repair — v3_psr_repair_f1fix.log (training log excerpt)

This is the V3 training log that produced the +4608 post_gelu activations cited
in the paper. Already committed at `logs/v3_psr_repair_f1fix.log` (commit
`8f9d12fea`, 254 lines). Values vary 4448-4864 across steps (single-run
snapshot, not a converged measurement).

**V4 is the production source.** V4 launched 2026-07-07 19:00 on RTX 3060 with
all F-1 fixes (KENDALL_FIXED_WEIGHTS=1, USE_PSR_TRANSITION=False, ablation_psr_only
preset). Log: `/tmp/train_psr_v4.log` (workstation-local, 1.4GB buffer headroom).

---

## Other checkpoints of record

| Checkpoint | Purpose | Source path |
|---|---|---|
| `crash_recovery.pth` | Resume point for V4 (epoch 30 state of V3) | `src/runs/rf_stages/checkpoints/crash_recovery.pth` |
| `d1_yolov8m_v3/weights/best.pt` | Single-task YOLOv8m detector (D1R 0.995) | `src/runs/d1_yolov8m_v3/weights/best.pt` |
| `activity_mvit_probe/frozen_probe.pth` | Frozen MViTv2-S activity probe (0.3810) | `src/runs/rf_stages/checkpoints/activity_mvit_probe/` |
| `d3_full_38k/best.pth` | Multi-task det (D3 0.00009 broken) | `src/runs/d3_full_38k/` |

(For these, see the per-folder `metrics.json` or `results.json` for the
training run that produced them. The hashes for non-`best.pth` files are
intentionally not enumerated here — they're large binaries that already
live under git-lfs-or-equivalent tracking, and the per-folder metrics
files are the source of record for their reported numbers.)
