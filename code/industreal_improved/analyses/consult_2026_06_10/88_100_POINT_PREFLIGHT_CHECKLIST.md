# 88 — 100-Point Pre-Flight Checklist [2026-07-02]

**Status:** All 100 items verified.
**Probe result:** 50-step probe completed (see file 87). All critical issues found and fixed (commit `75a2fe2`).
**Next:** `bash scripts/run_rf4_probe.sh` to confirm fixes, then RF4 full launch.

---

## A. Crash Hardening (Items 1-15)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 1 | CUDA_LAUNCH_BLOCKING=1 set BEFORE torch import | ✅ | `train.py:16` — 77 lines before `import torch` |
| 2 | CUDA_MODULE_LOADING=LAZY set before import | ✅ | `train.py:25` |
| 3 | NVIDIA_TF32_OVERRIDE=0 set before import | ✅ | `train.py:21` |
| 4 | CUDNN_DETERMINISTIC=True, BENCHMARK=False | ✅ | `train.py:296-297` |
| 5 | SIGHUP handler registered (does NOT exit) | ✅ | `train.py:1083-1093` — logs warning and returns |
| 6 | SIGTERM/SIGINT handler saves crash_recovery then sys.exit(0) | ✅ | `train.py:1065-1077` |
| 7 | BaseException catch in retry loop (not just Exception) | ✅ | `train.py:4284` — catches SystemExit from signal handler |
| 8 | OOM recovery: halve batch_size, rebuild dataloader | ✅ | `train.py:4335-4369` |
| 9 | DataLoader worker ENOMEM fallback to num_workers=0 | ✅ | `train.py:4311-4333` |
| 10 | Maximum retry limit (6 attempts) | ✅ | `train.py:4256` |
| 11 | Scheduler state saved in crash_recovery.pth | ✅ | `train.py:870-885` (FIX in commit ba8c4d2) |
| 12 | Watchdog timeout=1200s with PID matching guard | ✅ | `train.py:4092-4103` |
| 13 | Watchdog configurable via WATCHDOG_TIMEOUT | ✅ | `config.py:537` |
| 14 | Crash checkpoint saved before each epoch and validation | ✅ | `train.py:1104, 4661` |
| 15 | Mid-epoch resume (fast-forward DataLoader) | ✅ | `train.py:3873-3882` |

---

## B. CUDA/GPU Stability (Items 16-25)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 16 | GradScaler enabled for mixed precision | ✅ | `train.py:3727` — `enabled=C.MIXED_PRECISION` |
| 17 | FAULT_HANDLER for segfaults | ✅ | `train.py:33-36` |
| 18 | PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True | ✅ | `train.py:8` |
| 19 | CUBLAS_WORKSPACE_CONFIG set | ✅ | `train.py:25` (hardened) |
| 20 | MALLOC_ARENA_MAX=4 (glibc memory fragmentation) | ✅ | `train.py:109` |
| 21 | GPU heartbeat file updated every 100 steps | ✅ | `train.py:2112-2122` |
| 22 | ThreadPoolExecutor timeout for eval hang detection | ✅ | `train.py:4700-4711` |
| 23 | GAP-B segment metrics skip when SIGALRM unavailable | ✅ | `evaluate.py:3599` (FIX in 75a2fe2) |
| 24 | Zombie CUDA context cleanup on eval timeout | ✅ | `train.py:4796-4814` |
| 25 | Seg_timeout reduced to 240s (from 600s) | ✅ | `evaluate.py:3585` |

---

## C. Configuration Integrity (Items 26-35)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 26 | All presets have correct detach_reg_fpn=False | ✅ | 12/12 RF presets checked |
| 27 | All presets have correct DET_GT_FRAME_FRACTION | ✅ | 0.40 for RF3+, 0.90 for det-only, 0.0 for no-det |
| 28 | WEIGHT_DECAY=1e-3 (corrected from 5e-2) | ✅ | `config.py:528` |
| 29 | GRAD_CLIP_NORM=5.0 (corrected from 1.0) | ✅ | `config.py:541` |
| 30 | OneCycleLR steps_per_epoch=1 | ✅ | `train.py:3706` |
| 31 | SeqLR: [LinearLR(2ep), OneCycleLR(pct_start=0.1)] | ✅ | `train.py:3702-3703` |
| 32 | max_lr list length matches param group count | ✅ | 8 or 9 entries — 1:1 match |
| 33 | ACT_CLASS_GROUPING='hybrid' (no OOM at import) | ✅ | `config.py:303` + lightweight CSV counting (75a2fe2) |
| 34 | NUM_ACT_OUTPUTS lazy-loaded (not at module import) | ✅ | `config.py:1966` via `_act_grouping()` |
| 35 | WATCHDOG_TIMEOUT=1200 in config | ✅ | `config.py:537` |

---

## D. Detection Head (Items 36-47)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 36 | OHEM active: ratio=2.0, min_neg=32 | ✅ | `config.py:684, 687` |
| 37 | Asymmetric gamma: pos=0.0, neg=1.5 | ✅ | `config.py:697, 698` |
| 38 | DET_GT_FRAME_FRACTION=0.40 redistributes batch mass | ✅ | `config.py:832, dataset.py:1468-1511` |
| 39 | Empty-frame bg loss: 2048 samples, scale=0.05 | ✅ | `config.py:795, 796` |
| 40 | REINIT_REG_WARMUP: 1000 steps, 1%→100% | ✅ | `config.py:909, 910` |
| 41 | GIoU weight=2.0 in loss composition | ✅ | `config.py:676, losses.py:1235` |
| 42 | Bias init: pi=0.01 → bias=-4.60 (reinit), pi=0.03→-3.48 (normal) | ✅ | `model.py:542` |
| 43 | DET_EVAL_SCORE_THRESH=0.001 | ✅ | `config.py:610` |
| 44 | COCO-style mAP@0.5 with 101-point PR interpolation | ✅ | `evaluate.py:1368-1374` |
| 45 | cls_preds mean logged every 500 steps in [DET-HEALTH] | ✅ | `train.py:1432+` |
| 46 | det_gt_fraction logged every 500 steps | ✅ | `train.py:1510` (FIX in 2e69b1e) |
| 47 | All 4 collapse mechanisms redundant: any 3 of 4 fail-safe | ✅ | Verified by agent analysis |

---

## E. Activity Head (Items 48-60)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 48 | ACTIVITY_HEAD_SIMPLE=True (TCN=None, ViT=None) | ✅ | `config.py:865, model.py:1373-1377` |
| 49 | Simple MLP: 512→256→N (150K params) | ✅ | `model.py:1378-1401` |
| 50 | Hybrid grouping: ≥100 frames standalone, <100 verb-grouped | ✅ | `config.py:344-403` |
| 51 | Index 0 reserved for 'other' (unknown/zero-frame verbs) | ✅ | `config.py:374` |
| 52 | Label remap at all 4 production sites — no double-remap | ✅ | dataset.py:796-804, 904-908, 1020-1023; train.py:3427-3448 |
| 53 | Balanced sampler with floor=15 | ✅ | `dataset.py:1427-1431, config.py:718-719` |
| 54 | 3-layer sampler: balanced → task-aware → GT-frac | ✅ | `dataset.py:1427-1511` |
| 55 | CE + CB weights (NOT CB-Focal) | ✅ | `losses.py:1063-1066, 1125-1135, config.py:727` |
| 56 | Activity ramp: (epoch+1)/5 over 5 epochs | ✅ | `losses.py:1386-1390, config.py:755` |
| 57 | Stage-local ramp counter (_act_epoch_counter) | ✅ | `losses.py:1171-1187` |
| 58 | pred_distinct and entropy logged in [DIVERSITY] | ✅ | `evaluate.py:3491-3497` |
| 59 | ACT_LOSS_CAP=80.0 (safety net, never binds for CE) | ✅ | `config.py:756` |
| 60 | Segment eval correctly remaps labels (Q5 fix) | ✅ | `evaluate.py:875-877` |

---

## F. PSR Head (Items 61-70)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 61 | Binary focal loss active (gamma=0.5, alpha=0.25) | ✅ | `losses.py:1072, config.py:965, 958` |
| 62 | Per-component alpha: 2*(1-prevalence), clamp min=0.1 | ✅ | `losses.py:1156-1163` |
| 63 | PSR_COMP_WEIGHTS multiplicative with alpha | ✅ | `losses.py:896-902` |
| 64 | Step-based warmup: 500 steps, 2.0→1.0 | ✅ | `losses.py:1796-1800` |
| 65 | Epoch-based warmup: 3 epochs (staged path) | ✅ | `config.py:929` |
| 66 | Sequence mode: T=8 every 2 batches | ✅ | `config.py:978, 981` |
| 67 | Sequence DataLoader NOT shuffled (is_train=False) | ✅ | `train.py:3394-3413` — split='train_seq' |
| 68 | CausalTransformer has upper-triangular mask | ✅ | `model.py:2098-2099` |
| 69 | psr_comp_acc computed at validation | ✅ | `evaluate.py:3761` (FIX in 2e69b1e) |
| 70 | PSR per-frame component recognition (NOT transition) | ✅ | Confirmed: binary sigmoid [B,11] output per frame |

---

## G. Head Pose (Items 71-78)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 71 | Real GT from pose.csv (NOT body keypoints) | ✅ | `dataset.py:576, config.py:41-42` |
| 72 | HEAD_POSE_POS_SCALE=100 applied at dataset load | ✅ | `dataset.py:598-599, config.py:765` |
| 73 | HeadPoseHead: GAP(C4||C5)→1152→512→256→9 | ✅ | `model.py:1498-1512` |
| 74 | forward_angular_MAE_deg via proper arccos(normalized) | ✅ | `evaluate.py:1816-1821` |
| 75 | GT-norm removed from unit-vector guard | ✅ | `evaluate.py:1840-1841` (FIX in 75a2fe2) |
| 76 | KENDALL_HP_PREC_CAP: lv_hp = max(lv_hp, lv_det.detach()) | ✅ | `losses.py:1689-1690` |
| 77 | log_var_pose SHARED between body + head pose (no double-count) | ✅ | `losses.py:1766-1780` |
| 78 | Body pose: pseudo-keypoints from boxes, DO NOT report | ✅ | `config.py:406-411` |

---

## H. Multi-Task Kendall (Items 79-86)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 79 | All 4 log_vars initialized correctly | ✅ | `losses.py:1027-1035` |
| 80 | Triple NaN guards (before cap, before Kendall, inside Kendall) | ✅ | `losses.py:1272-1302, 1579-1594, 1804-1827` |
| 81 | Smooth caps on all 5 losses | ✅ | det=50, pose=30, act=80, psr=20, hp=30 |
| 82 | Kendall bounds: ACT min=-0.5, PSR max=0.0, POSE max=3.0 | ✅ | `config.py:888-890` |
| 83 | Task group assembly: 4 heads, correct order, lv_hp counted once | ✅ | `losses.py:1752-1801` |
| 84 | NaN guards include clamp + 0.0 * log_vars trick | ✅ | `losses.py:1818-1827` |
| 85 | PSR step warmup applies to prec_psr (not loss_psr) | ✅ | `losses.py:1796-1800` |
| 86 | Both weighting mechanisms for PSR simultaneously active | ✅ | alpha * focal * comp_weights — multiplicative |

---

## I. Data Pipeline (Items 87-93)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 87 | Sequence windows built stride=1 (not strided) | ✅ | `dataset.py:1257` — stride=1 in _build_seq_sample_index |
| 88 | DET_GT_FRAME_FRACTION redistributes correctly (within-pool) | ✅ | `dataset.py:1499-1500` |
| 89 | Per-class sampling mass diagnostic logged | ✅ | `dataset.py:1519-1538` |
| 90 | collate_fn handles variable-length detection boxes | ✅ | `dataset.py:1602-1617` |
| 91 | Sequence collate includes activity_mask | ✅ | `dataset.py:1743` |
| 92 | NUM_WORKERS=0 (deadlock workaround) | ✅ | `config.py:551-554` |
| 93 | Pose.csv loading does NOT fail on missing file (zeros) | ✅ | `dataset.py:577-578` |

---

## J. Evaluation Integrity (Items 94-100)

| # | Check | Pass | Evidence |
|---|-------|------|----------|
| 94 | EVAL_MAX_BATCHES=10 correctly enforced on all paths | ✅ | main, ThreadPool, subprocess (FIX in 75a2fe2 for subprocess edge case) |
| 95 | EVAL_MAX_BATCHES=0 means "unlimited" (no silent 2500 cap) | ✅ | `subprocess_eval.py:113, 122` (FIX in 75a2fe2) |
| 96 | Activity macro-F1 uses present_labels filter | ✅ | `evaluate.py:956-958` |
| 97 | Detection mAP uses present-class and standard | ✅ | `evaluate.py:1665` — `det_mAP50_pc` |
| 98 | Head pose position MAE unit warning present | ✅ | `evaluate.py:1861-1866` |
| 99 | Segment eval skipped when SIGALRM unavailable (thread safety) | ✅ | `evaluate.py:3599` (FIX in 75a2fe2) |
| 100 | Subprocess eval JSON serialization with clean() helper | ✅ | `subprocess_eval.py:126-135` |

---

## Known Remaining Issues (Non-Blocking)

| Issue | Impact | When to Fix |
|-------|--------|-------------|
| LR discontinuity at epoch 2 (27.5x drop from warmup to OneCycleLR) | Wastes ~5 epochs of re-warmup | Post-RF4 |
| Sequence path bypasses RAM cache (opens PIL directly) | ~50ms/frame disk I/O for sequence batches | Performance only |
| norm_regularizer in head_pose_loss_split disabled (dead code) | No effect — both losses already normalize internally | Post-RF4 |
| Activity ramp double-ramp when STAGED_TRAINING=True (currently False) | Latent bug only in staged mode | Post-RF4 |
| Detection/activity paper targets may be inflated vs internal analysis | Paper framing issue, not code | Camera-ready stage |
| Head pose position MAE units unverified | DO NOT report | Camera-ready stage |
