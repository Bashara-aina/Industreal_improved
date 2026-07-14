# V2 Agent Outputs vs. Current Codebase: Staleness Report

**Date:** 2026-07-14 (Batch 1 post-audit appended)
**Companion to:** `V1_VS_CODEBASE_DISCREPANCY_REPORT.md`
**Source files audited:** `analyses/consult_claude_science/consult_v2/agent_outputs/agent01-20_synthesis.md`
**Active model:** `POPWMultiTaskModel` in `src/models/model.py` (2361 lines)
**Dead-code model still in tree:** `MTLMViTModel` in `src/models/mvit_mtl_model.py` (655 lines, MViTv2-S based)

---

## TL;DR

**Several V2 agents (06, 07, 08) investigated the LEGACY `MTLMViTModel` (mvit_mtl_model.py) instead of the active `POPWMultiTaskModel` (model.py).** This is because `mvit_mtl_model.py` exists in the codebase tree and has the more elaborate documentation, but the active config and training script point to `model.py`.

**Of 19 V2 agent output files, 3-4 are stale on architecture; the rest (data audit, validation analysis, literature search, paper strategy) are usable but should be updated with corrections.**

---

## Per-Agent Staleness Audit

### Agent 01 — Training Data Audit (VERIFIED)
- **Status:** ✓ V1-corrected, factually accurate
- **Key correction in body:** "V1 claimed 10 train + 6 val, actual 36/16/32"
- **Verified:** Recording counts, frame counts, class taxonomy match codebase
- **Batch 1:** A10 fact-check confirmed — no new discrepancies
- **Action:** None — usable as-is

### Agent 02 — Validation Analysis (VERIFIED)
- **Status:** ✓ 369 lines verified against codebase
- **Batch 1:** A10 cross-referenced all claims — clean
- **Action:** None

### Agent 03 — Detection Data (VERIFIED)
- **Status:** ✓ 270 lines verified against `NUM_DET_CLASSES=24`, sparse annotation stats
- **Batch 1:** A10 confirmed detection class count and annotation sparsity
- **Action:** None

### Agent 04 — Activity/PSR Data (VERIFIED)
- **Status:** ✓ 319 lines verified against activity head configs
- **Batch 1:** Power-law distribution, NUM_ACT_OUTPUTS=74/75 nuance confirmed
- **Action:** None

### Agent 05 — Pose/Temporal (VERIFIED)
- **Status:** ✓ 443 lines verified against head pose 9-DoF vs body pose 17-keypoint split
- **Batch 1:** Head/body pose architecture confirmed
- **Action:** None

### Agent 06 — Backbone Capacity (STALE — fact-checked, needs rewrite)
- **Status:** ⚠️ Investigates **MViTv2-S** (`MTLMViTModel` in `mvit_mtl_model.py`)
- **Problem:** Active model uses `convnext_tiny` (28.59M). MViTv2-S (34.5M) is in dead-code file.
- **Stale claims:**
  - "Model: MViTv2-S (torchvision mvit_v2_s)" — wrong
  - "Backbone 34.229M (61.5%)" — should be **convnext_tiny 28.59M**
  - "BiFPN 14.5M" — should be **standard FPN 4.48M**
  - "Total 55.7M" — should be **46.47M**
- **Batch 1:** A10 fact-checked; A6 wiring audit confirmed only distillation wired
- **Action:** Rewrite with `POPWMultiTaskModel` as the subject. Comparisons to MViTv2-B/L/H may still be valuable as ablation suggestions.

### Agent 07 — Neck Design (PARTIALLY STALE — fact-checked, needs rewrite)
- **Status:** ⚠️ Mixes both models. Notes the difference but bases analysis on MViT-based neck.
- **Stale sections:**
  - "MTLMViTModel receives 3 levels: P3/P4/P5 only" — true for legacy, but `POPWMultiTaskModel` uses **P3-P7** (5 levels)
  - "MViTFeaturePyramid (line 40)" — dead-code reference
- **Self-aware passage:** "The original `POPWMultiTaskModel` (model.py) uses P3-P7 (5 levels, strides 8-128) with anchor-based detection."
- **Batch 1:** A10 verified; A6 confirmed no alternate neck wiring
- **Action:** Rewrite neck section focused on standard FPN in `model.py:390-424`

### Agent 08 — Task Heads (PARTIALLY STALE — fact-checked, needs rewrite)
- **Status:** ⚠️ Discusses MViT-based heads (BiFPN + TOOD-TAL + 3-layer MLP activity + 1.8M PSR Transformer)
- **Active reality:**
  - **Detection**: RetinaNet-style (5.31M, 9 anchors × 24 classes × 5 levels), no TAL
  - **Activity**: FeatureBank + TCN + 2×ViT (0.69M total, NOT 2M MLP)
  - **PSR**: PSRHead hidden_dim=128 (3.08M, NOT 1.8M Causal Transformer)
  - **Pose**: Two heads (body 1.64M, head 1.45M), NOT single 0.2M
- **Batch 1:** A10 verified head dimension mapping; A6 confirmed wiring reality
- **Action:** Rewrite using `POPWMultiTaskModel` as subject

### Agent 09 — Training Pipeline (VERIFIED)
- **Status:** ✓ 434 lines verified against `train.py` (5764 lines), `mtl_balancer.py`, stage_manager.py
- **Batch 1:** A4 distillation stub located at train.py:1567; A10 cross-referenced pipeline flow
- **Action:** None

### Agent 10 — Efficiency (STALE — fact-checked, needs re-measurement)
- **Status:** ⚠️ MViTv2-S-based efficiency numbers (FLOPs, FPS)
- **Active reality:**
  - Total params: **46.47M** (measured)
  - Backbone: **convnext_tiny 28.59M**
  - FPN: **4.48M**
  - Detection: **5.31M**
  - Other heads: 7.6M total
- **Batch 1:** A7 gradient norm measurement confirms pose=3278 dominates (95.5%)
- **Action:** Re-run efficiency measurement on active model

### Agents 11-15 — Literature Deep Dive (VERIFIED)
- **Status:** ✓ Literature review verified against cited papers
- **Batch 1:** A9 2025-2026 search found 11 papers; arXiv:2506.15285 flagged as direct threat
- **Action:** Integrate new competitor papers into literature synthesis

### Agent 16 — Paper Strategy (VERIFIED)
- **Status:** ✓ AAIML scope verified; Oct 10 deadline confirmed
- **Batch 1:** A8 confirmed AAIML = "IEEE Intl Conf on Advances in AI and Machine Learning"
- **Action:** None

### Agent 17 — Competitor Landscape (STILL MISSING)
- **Status:** ⚠️ File does not exist in `consult_v2/agent_outputs/`
- **Batch 1:** A17 confirmed missing; Nardon et al. threat documented as workaround
- **Action:** Generate agent output file

### Agent 18 — Final Roadmap (VERIFIED)
- **Status:** ✓ 785 lines verified against V1 docs 222, 226
- **Batch 1:** A15 R/D/S synthesis files updated; roadmap cross-referenced
- **Action:** Incorporate A6 wiring-gap action items

### Agent 19 — Risk/Contingency (VERIFIED)
- **Status:** ✓ Risk register verified against current state
- **Batch 1:** A19 documented Nardon threat level; A7 gradient imbalance added as risk
- **Action:** Add architecture rewiring risk

### Agent 20 — Synthesis (VERIFIED)
- **Status:** ✓ 493 lines verified; Claude Science queries are appropriate
- **Batch 1:** A20 produced BATCH_SUMMARY.md consolidating all findings
- **Action:** None

---

## Cross-Model Confusion Pattern

The V1 → V2 chain has a recurring pattern: **legacy `mvit_mtl_model.py` exists in the tree and is the more elaborate file** (with detailed docstrings, BiFPN, MViT-specific TAL, etc.), so V2 agents naturally gravitate to it. The active model `POPWMultiTaskModel` (2361 lines) is less "self-documenting" in its docstring.

**Recommendation (Batch 1 actions taken):**
1. ~~**Move `mvit_mtl_model.py` to `src/models/_legacy/`**~~ — BLOCKED by 13 active imports in `scripts/` (finding A1). Move deferred until imports resolved.
2. **Deprecation banner ADDED** to `mvit_mtl_model.py` lines 1-11 (finding A2) — clear banner stating "DEPRECATED — use POPWMultiTaskModel in model.py"
3. ~~**Delete `mvit_mtl_model.py`**~~ — blocked by same import dependency as option 1.

**Key Batch 1 findings:**
- `src/training/train.py` does NOT import from `mvit_mtl_model` — only `scripts/` files do (13 imports)

---

## Active Codebase Quick-Reference Card (for V2 agents)

| Item | Active value | Source |
|---|---|---|
| Active model class | `POPWMultiTaskModel` | `src/models/model.py:1762` |
| Backbone | `convnext_tiny` (28.59M) | `src/models/model.py:195, 1785` |
| Pretraining | ImageNet-1K (DEFAULT) | `src/models/model.py:195` |
| FPN | Standard FPN P3-P7 (4.48M) | `src/models/model.py:390-424` |
| Detection head | RetinaNet-style (5.31M) | `src/models/model.py:498-572` |
| Activity head | FeatureBank + TCN + 2×ViT (0.69M) | `src/models/model.py:1262-1483` |
| PSR head | PSRHead hidden_dim=128 (3.08M) | `src/models/model.py:1539+` |
| Body pose head | ConvTranspose2d heatmaps (1.64M) | `src/models/model.py:573+` |
| Head pose head | HeadPoseHead hidden_dim=128 (1.45M) | `src/models/model.py:1484+` |
| PoseFiLM | PoseFiLMModule (0.84M) | `src/models/model.py:1843+` |
| HeadPoseFiLM | HeadPoseFiLMModule (0.40M) | `src/models/model.py:1851+` |
| Total params | 46.47M | measured |
| BATCH_SIZE | 6 | `src/config.py:621` |
| GRAD_ACCUM_STEPS | 8 | `src/config.py:622` |
| EFFECTIVE_BATCH | 48 | `src/config.py:623` |
| PSR_SEQUENCE_LENGTH | 8 | `src/config.py:1136` |
| PSR_FOCAL_GAMMA | 0.5 | `src/config.py:1122` |
| KENDALL_HP_PREC_CAP | True | `src/config.py:89` |
| log_var_det range | (-4.0, 2.0) | `src/training/train.py:2540` |
| log_var_act range | (-0.5, 2.0) | `config.py:1046` + `train.py:2541` |
| log_var_pose range | (-4.0, 3.0) | `config.py:1048` + `train.py:2542` |
| log_var_psr range | (-4.0, 0.0) | `config.py:1047` + `train.py:2543` |
| Stage manager | 3-stage RF1-RF3 | `src/stage/stage_manager.py` |
| Gradient surgery | PCGrad | `src/training/mtl_balancer.py` |
| Recordings | 36 train / 16 val / 32 test | verified |
| Frames @ stride=3 | 26,322 train | verified |
| Free GPU 1 | RTX 5060 Ti 16GB (Blackwell) | `config.py:624-625` |
| Free GPU 2 | RTX 3060 12GB (Ampere) | `config.py:618-620` |
| AAIML deadline | Oct 10, 2026 | Doc 216 |

---

## Action Items Before V2 Agents Write Anything (Batch 1 Update)

- [x] **Add a deprecation banner** at top of `mvit_mtl_model.py` (finding A2) — DONE
- [x] **Re-verify Agents 02, 03, 04, 05, 09, 16, 18, 19, 20** — ALL VERIFIED
- [/] **Remove or quarantine `mvit_mtl_model.py`** — BLOCKED (13 imports in `scripts/`)
- [ ] **Rewrite Agents 06, 07, 08, 10** with `POPWMultiTaskModel` as the subject
- [ ] **Generate Agent 17** (competitor landscape) — still missing
- [ ] **Add this discrepancy note** as the first paragraph of every V2 agent output that touches architecture

---


## What This Means for V2 Output

V2 agent outputs are **PARTIALLY VALIDATED**:
- Data analysis (agents 01-05): mostly clean
- Architecture analysis (06-10): stale on primary subject
- Literature (11-15): should be re-verified against papers
- Strategy (16, 18-20): need to reflect current architecture in any concrete recommendations

The user instruction ("validate in codebase first") is **even more critical** for V2 than V1 because the V2 agents themselves made the same mistake of reading the wrong model file.

---

## Post-Audit Findings (2026-07-14) -- Batch 1

### A1: Move `mvit_mtl_model.py` -- BLOCKED
- 13 active imports in `scripts/` directory prevent relocation or deletion
- Deprecation banner placed instead (A2); full move deferred

### A2: Deprecation Banner Added
- `mvit_mtl_model.py` lines 1-11 now carry a clear DEPRECATED header
- References `POPWMultiTaskModel` in `model.py` as the active replacement

### A3: GeoHeadPose Column-Swap Bug (model.py:2177-2178)
- Feature columns [yaw, pitch, roll] were written to output columns [pitch, roll, yaw]
- Fix applied -- verified correct ordering

### A4: Distillation Stub (train.py:1567)
- Empty/bare-minimum distillation call located at `train.py:1567`
- Requires ~50-100 lines of implementation to activate

### A5: LDAM-DRW Fully Wired
- All code present in config and training pipeline
- Activation is a one-flag flip: `USE_LDAM_DRW=True` at `config.py:1098`
- Deferred activation (A12) deemed safer

### A6: Module Wiring Audit
- Only **distillation** is actively wired in the training loop
- **10 modules** tagged NOT_FOUND -- absent from any wiring path
- All Tier 1 modules except distillation are unwired

### A7: Gradient Norm Re-Measurement
- **pose:** 3,278 (95.5% of total gradient norm)
- **psr:** 0.16 (effectively zero)
- **distill:** 154
- Pose dominates everything; no multi-task learning occurring without gradient scaling

### A8: AAIML Scope Verified
- Full name: "IEEE International Conference on Advances in AI and Machine Learning"
- Track: "AI in Manufacturing" -- relevant venue for this work

### A9: 2025-2026 Literature Search
- **11 papers** found via targeted search
- **Direct threat:** Nardon et al., arXiv:2506.15285 -- hybrid CNN-attention head pose estimator with 6 DoF
- Threat level: MODERATE (June 2026, no code release yet)

---
