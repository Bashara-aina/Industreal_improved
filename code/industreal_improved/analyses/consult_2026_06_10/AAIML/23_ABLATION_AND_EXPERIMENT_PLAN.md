# AAIML 2027 — Ablation & Experiment Plan for Camera-Ready

**Goal:** All experiments must be completed by October 1, 2026 (9 days before deadline).
**GPU Budget:** Two RTX 3060 12GB (GPU 0 for ablations, GPU 1 for primary).

---

## Critical Path Ablations (Must Complete — Validate All 3 Pathologies)

These 3 ablations directly validate the paper's core claims and must be prioritized.

### Ablation 1: Simple MLP vs TCN/ViT (Validates Pathology 1 + Fix)

| Aspect | Detail |
|--------|--------|
| **Config change** | `config.py:861` — toggle `ACTIVITY_HEAD_SIMPLE = True → False` |
| **Code reference** | `models/model.py:1319-1422` — conditional TCN+ViT allocation |
| **Conditions** | 2: MLP (150K) vs TCN+2xViT (8.2M) |
| **Epochs needed** | 15 (collapse happens in first 5-10 epochs) |
| **GPU hours** | ~35 hours (2 × 15 epochs on secondary GPU) |
| **Expected result** | MLP: ~X% top-1. TCN+ViT: <3% top-1 (collapse), pred entropy <0.1 |
| **Paper value** | Directly validates the central thesis of Pathology 1 |

### Ablation 2: Balanced vs CB Sampling (Validates Pathology 1 Mechanism)

| Aspect | Detail |
|--------|--------|
| **Config change** | `config.py:714` — toggle `ACT_SAMPLER_MODE = 'balanced' → 'cb'` |
| **Code reference** | `industreal_dataset.py:1427-1439` — sampler logic |
| **Conditions** | 2: 'balanced' (current) vs 'cb' (legacy beta=0.99) |
| **Epochs needed** | 15 (sampler imbalance causes collapse in first 5-10 epochs) |
| **GPU hours** | ~35 hours |
| **Expected result** | Balanced: stable training. CB: activity collapse to 2-3 classes |
| **Paper value** | Shows Pathology 1 is a sampler problem, not just a temporal-architecture problem |

### Ablation 3: Kendall Bounds (Validates Pathology 2 + Fix)

| Aspect | Detail |
|--------|--------|
| **Config change** | `config.py:888-890` — change `KENDALL_LOG_VAR_MIN_ACT` and related bounds |
| **Code reference** | `losses.py:1034, 1676-1682` — log_var init and clamping |
| **Conditions** | 3: (A) default [-4,2] with init 0, (B) per-task bounds [-0.5, 2.0] with init 0, (C) fixed weights (KENDALL_FIXED_WEIGHTS=True) |
| **Epochs needed** | 18 (log_var dynamics converge within 15 epochs) |
| **GPU hours** | ~55 hours (3 × 18 epochs) |
| **Expected result** | (A) s_act reaches -4 by ~epoch 15, weight drops to 0.018. (B) s_act stabilizes at -0.5 ± 0.3. (C) fixed weights give stable but suboptimal balancing |
| **Paper value** | Directly validates the Pathology 2 fix mechanism. s_act trajectories are the key figure. |

**Total critical path:** ~125 GPU hours = ~5.2 days on one GPU = ~2.6 days with two GPUs.

---

## Supplementary Ablations (If GPU Time Permits)

### Ablation 4: DET_GT_FRAME_FRACTION (Supports Pathology 2)

| Aspect | Detail |
|--------|--------|
| **Config change** | Env var `DET_GT_FRAME_FRACTION=0.40,0.60,0.90,0.0` at `config.py:828` |
| **Code reference** | `industreal_dataset.py:1468-1511` |
| **Conditions** | 4 |
| **Epochs needed** | 20 (detection mAP50 needs 20+ epochs) |
| **GPU hours** | ~90 hours |
| **Paper value** | Shows that 0.90 starves activity (0.14 frames/class/batch); 0.40 preserves multi-task balance |

### Ablation 5: MLP vs TCN/ViT at T=4 vs T=16 (2x2 sub-grid for Figure 3)

| Aspect | Detail |
|--------|--------|
| **Config change** | `config.py:861` + `models/model.py:1864,1878` (window_size hardcoded) |
| **Code reference** | FeatureBank T accumulation |
| **Conditions** | 4: MLP×T=4, MLP×T=16, TCN×T=4, TCN×T=16 |
| **Epochs needed** | 15 |
| **GPU hours** | ~90 hours |
| **Paper value** | Replaces the full 4×4 heatmap with a tractable subset that still shows the collapse boundary |

### Ablation 6: GRAD_CLIP_NORM 1.0 vs 5.0 (Supports Pathology 3)

| Aspect | Detail |
|--------|--------|
| **Config change** | `config.py:537` — toggle `GRAD_CLIP_NORM = 5.0 → 1.0` |
| **Conditions** | 2 |
| **Epochs needed** | 20 |
| **GPU hours** | ~45 hours |

### Ablation 7: WEIGHT_DECAY 1e-3 vs 5e-2

| Aspect | Detail |
|--------|--------|
| **Config change** | `config.py:528` — toggle `WEIGHT_DECAY = 0.001 → 0.05` |
| **Conditions** | 2 |
| **Epochs needed** | 20 |
| **GPU hours** | ~45 hours |

---

## Primary Training Run (Must Be Completed First)

### RF1-RF10 Full Protocol

| Stage | Epochs | Data | What Happens |
|-------|--------|------|-------------|
| RF1 | 20 | 20% | Detection bootstrap (reinit heads) |
| RF2 | 15 | 35% | Add head pose |
| RF3 | 15 | 35% | Add activity (ramp epochs 0-4) |
| RF4 | 20 | 50% | **First full 4-task** |
| RF5 | 10 | 50% | Continued training |
| RF6 | 10 | 65% | Continued |
| RF7 | 10 | 65% | Continued |
| RF8 | 10 | 80% | Continued |
| RF9 | 10 | 90% | Continued |
| RF10 | 15 | 100% | **Final convergence** |
| **Total** | **135** | — | |

**GPU hours (primary):** ~120 hours for 135 epochs on RTX 3060.
**Start date:** Immediate. Must complete before ablations.

### Three-Seed Variance

After RF10 completes with seed 42:
- Run seeds 73 and 128 (change `config.py:561` — `C.SEED = 73` or `128`)
- Each: ~120 GPU hours
- **Total: ~360 GPU hours** for all 3 seeds
- With 2 GPUs running in parallel: ~7.5 days

### Metric Tracking

Log every epoch:
- Detection: mAP50 (present-class and standard), cls_mean, det_gt_fraction, score_max
- Activity: pred_distinct (#groups predicted), entropy, top-1 (clip-level), macro-F1 (present_labels)
- PSR: comp_acc (binary sigmoid accuracy), overall F1
- Head pose: forward_angular_MAE_deg, up_angular_MAE_deg (for disclosure)
- Kendall: lv_det, lv_pose, lv_act, lv_psr (trajectories over epochs)
- Scheduler: current LR value at each epoch

---

## Expected Results Table (Camera-Ready Target)

| Metric | Expected Range | Compared To |
|--------|---------------|-------------|
| Detection mAP50 (present-class) | 0.30-0.55 | YOLOv8m 0.838 (same ASD protocol) |
| Activity top-1 (47-group, clip-level) | 0.35-0.55 | MViTv2 0.653 (74-class — different task) |
| Activity macro-F1 (47-group) | 0.25-0.45 | Own baseline (first reported) |
| PSR comp-acc (per-frame) | 0.70-0.85 | Own baseline |
| PSR overall F1 | 0.55-0.75 | Own baseline |
| Head pose forward-gaze MAE | 8-15° | First reported on IndustReal |
| Forward-gaze MAE baseline | 8.71° | Established by prior analysis (file 82) |
| Head pose up-vector MAE | ~95° | Do NOT report — unlearned |

**Note on comparisons:**
- Detection vs YOLOv8m: SAM evaluation protocol/classes (24 ASD classes, COCO mAP@0.5). Our mAP50 will be lower because our detection head trains from scratch in multi-task setting vs YOLOv8m's dedicated COCO pretraining. State this explicitly.
- Activity vs MViTv2: DIFFERENT TASK. MViTv2 reports 74-class clip-level top-1. Our 47-group results are not comparable. We must "establish own baseline" per file 82 guidance.
- PSR: B1/B2/B3 baselines measure procedure-order detection, not per-frame component recognition. Different task — do NOT compare.
- Head pose: No existing IndustReal baseline — first reported result.

---

## Timeline (101 Days to Deadline)

| Date | Milestone | Duration |
|------|-----------|----------|
| Jul 1-7 | RF1-RF10 primary training (seed 42) | 7 days |
| Jul 8-15 | Seeds 73, 128 training (2 GPUs) | 7 days |
| Jul 10-15 | Critical path ablations 1-3 (GPU 0) | 5 days |
| Jul 15-30 | Factory pilot (Phase 1) | 15 days |
| Jul 20-Aug 1 | Supplementary ablations 4-7 (if GPU available) | 10 days |
| Aug 1-Sep 1 | Generate all figures (8 total, prioritize 3 main figures) | 30 days |
| Aug 1-Sep 1 | Write full paper with actual numbers | 30 days |
| Sep 1-15 | Internal review, reviewer defense prep | 15 days |
| Sep 15-30 | Code release, supplementary material, survey publication | 15 days |
| Oct 1-10 | Final polish, format compliance check, submission | 9 days |

**Risk: Factory pilot not started. If pilot misses the Jul 15-Aug 1 window, strip pilot to 1 paragraph with "results forthcoming" and remove from contributions.**

---

## Figure Plan (3 Mandatory + 2 Nice-to-Have)

### Mandatory (Must Complete)

**Figure 1:** Pathology 1 mechanism diagram. Dual-panel: (A) Data pipeline cadence mismatch (sampler → bank). (B) Optimization pipeline cadence mismatch (OneCycleLR → training loop). Show intended vs actual flow with red X at interface. Caption: ~3 lines.

**Figure 2:** Pathology 1 ablation results. Multi-panel: (A) MLP vs TCN+ViT top-1 over epochs. (B) Balanced vs CB sampler top-1 over epochs. (C) s_act trajectories for [-4,2] vs [-0.5,2] bounds. (D) pred_distinct over epochs for each condition. This is the paper's primary empirical evidence.

**Figure 3:** Gradient artifact illustration. Two histograms: per-parameter norm ratios (312x apparent gap) vs head-level RMS gradients (3x actual gap). Caption: ~3 lines showing how dimensionality artifacts produce spurious conclusions.

### Nice-to-Have

**Figure 4:** LR schedule trajectory. Intended (super-convergence: rise 10 epochs, decay 90) vs actual (flat at near-minimum for 98 epochs before fix). This directly visualizes Pathology 4 / the broadened Pathology 1.

**Figure 5:** Deployment pipeline diagram (4-swimlane: Camera GPU → PSR → Solana → Dashboard). Only if blockchain content stays in main paper (trimmed version).

### Figures to Cut (from current draft)
- ~~Figure 1: End-to-end pipeline~~ → keep only if 5 is not created
- ~~Figure 6: Blockchain pipeline~~ → move to supplementary
- ~~Figure 7: Worker dashboard~~ → move to supplementary
