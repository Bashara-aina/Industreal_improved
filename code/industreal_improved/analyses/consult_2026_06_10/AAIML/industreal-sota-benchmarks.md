# IndustReal SOTA Benchmarks — Curated Reference for Our Paper

**Source:** WACV 2024, Schoonbeek et al., "IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos"
**Paper:** https://openaccess.thecvf.com/content/WACV2024/html/Schoonbeek_IndustReal_A_Dataset_for_Procedure_Step_Recognition_Handling_Execution_Errors_WACV_2024_paper.html
**GitHub:** https://github.com/TimSchoonbeek/IndustReal
**Dataset:** 27 participants, 84 egocentric recordings, HoloLens 2, construction-toy car assembly

> **Purpose:** This file selectively extracts only the benchmark numbers relevant to our 4-task model (detection, activity, PSR, pose). Other IndustReal tasks (depth, visible light, stereo ablation) are omitted for brevity.

---

## 1. Action Recognition (AR) — Activity Head

### Best configurations (Table 2, test set)

| Model | Pretrain | Modalities | Top-1% | Top-5% |
|---|---|---|---|---|
| SlowFast | Kinetics | RGB | 60.39 | 85.21 |
| **MViTv2-S** | **Kinetics** | **RGB** | **65.25** | **87.93** |
| MViTv2-S | Kinetics | RGB+VL+stereo | 66.45 | 88.43 |

- MViTv2 is the strongest AR baseline on IndustReal
- MECCANO pretraining underperforms Kinetics pretraining
- Our activity head uses verb-grouped 69 classes, NOT comparable directly to 75-class AR

### Activity metrics used in our model

| Metric | What it measures | Used in our model |
|---|---|---|
| macro_f1 | Per-class F1 averaged equally | ✅ Combined metric |
| top-5 accuracy | Correct class in top 5 | ✅ Logged |
| frame accuracy | Per-frame classification | ✅ Logged |
| pred_distinct | # of classes predicted | ✅ Diagnostic |
| entropy | Prediction diversity (nats) | ✅ Diagnostic |

---

## 2. Assembly State Detection (ASD) — Detection Head

### Best configurations (Table 3, test set)

| Training | mAP (annotated frames) | mAP (entire videos) |
|---|---|---|
| Synthetic only (100K Unity) | 0.573 | 0.341 |
| Real only (26.9K frames) | 0.753 | 0.553 |
| Synthetic → Real fine-tune | **0.779** | **0.575** |
| **Real + Synthetic** | **0.838** | **0.641** |

**Best YOLOv8-m:** COCO-pretrained, trained on real + synthetic data. mAP=0.838 annotated, 0.641 entire videos.

### Error state performance (best model)

| Metric | Value |
|---|---|
| False positive rate on error states | 65% |
| AP on error states | 0.23 |
| FPS (V100) | 178 fps |

### Detection metrics used in our model

| Metric | What it measures | Used in our model |
|---|---|---|
| mAP@0.5 | Detection accuracy at IoU=0.5 | ✅ Combined metric |
| mAP50_pc | Present-class only (honest) | ✅ Gate metric |
| n_present | # classes detected | ✅ Logged |
| dp_scores mean | Prediction confidence | ✅ DET_PROBE |
| bestIoU_max | Localization quality | ✅ DET_PROBE |

---

## 3. Procedure Step Recognition (PSR) — PSR Head

### Baseline implementations (from paper)

| Baseline | Method |
|---|---|
| B1 | Naive: every ASD state change = step completion |
| B2 | Confidence accumulation over time |
| B3 | B2 + procedural knowledge constraints |

Each has a synthetic-only variant (B1-S, B2-S, B3-S).

### PSR metrics: All recordings (Table 4)

| Baseline | POS | F1 | τ (s) |
|---|---|---|---|
| B1 | 0.570 | 0.779 | 14.9 |
| B2 | **0.731** | **0.860** | 22.3 |
| **B3 (best)** | **0.797** | **0.883** | **22.4** |

### PSR metrics: Recordings with errors only

| Baseline | POS | F1 | τ (s) |
|---|---|---|---|
| B1 | 0.480 | 0.698 | 14.4 |
| B2 | 0.636 | **0.784** | 20.2 |
| **B3 (best)** | **0.731** | **0.816** | **20.4** |

### Key insight for our paper

Our PSR head learns per-frame component recognition (11 binary states), NOT transition-based PSR like B2/B3. The B2/B3 baselines use ASD outputs + procedural knowledge to infer step completions. Ours is a fundamentally different approach — frame-level state estimation rather than transition detection.

The comparable baseline is **B1** (naive: every state change = step), but even B1 has access to a much stronger detection backbone (YOLOv8m at 0.838 mAP vs our 0.212 mAP).

**Honest framing:** "Our PSR head performs per-frame component state recognition as a byproduct of multi-task training, not transition detection. The B2/B3 baselines use specialized ASD + procedural knowledge pipelines and are not directly comparable."

### PSR metrics used in our model

| Metric | What it measures | Used in our model |
|---|---|---|
| psr_f1@±3 | Transition F1 (±3 frame tolerance) | ✅ Combined metric (F22 fix active) |
| POS | Procedure order similarity | Logged but currently 0 (eval bug) |
| Binary acc | Per-component accuracy | ✅ Diagnostic |
| unique patterns | Distinct state vectors predicted | ✅ Diagnostic |

---

## 4. Head Pose / Ego-Pose

**No prior baseline exists on IndustReal.** This is our contribution.

| Metric | Our value (epoch 5) | Prior work |
|---|---|---|
| Forward MAE | **8.92°** | None |
| Up MAE | **7.48°** | None |
| Position | ⚠️ Not reportable | N/A |

**Framing:** "We establish the first ego-pose baseline on IndustReal assembly data, achieving 8.92° forward MAE at zero additional inference cost as a byproduct of multi-task training."

---

## 5. Dataset Statistics (Context)

| Statistic | Value |
|---|---|
| Participants | 27 (12 train / 5 val / 10 test — subject split) |
| Recordings | 84 egocentric videos |
| Duration | ~5.8 hours |
| AR classes | 75 fine-grained action classes |
| AR instances | 9,273 annotated |
| ASD frames | 26.9K annotated frames (13%) |
| ASD classes | 22 correct states + 27 error states |
| ASD error frames | 3,569 |
| PSR correct completions | 724 (8.6 ± 1.2 per recording) |
| PSR errors | 38 (14 unseen in val/test) |

---

## 6. How Our Metrics Compare (Honest)

| Task | SOTA (WACV 2024) | Ours (epoch 5) | Comparable? |
|---|---|---|---|
| AR top-1 | 65.25% (MViTv2, 75-class) | — | ❌ Different class taxonomy (75 vs 69 verb-grouped) |
| ASD mAP@0.5 | 0.838 (YOLOv8m, real+synth) | **0.212** | ⚠️ Gap is 75% — but ours is multi-task byproduct at $299 GPU |
| ASD mAP50_pc | — | **0.339** | Present-class (more honest) |
| PSR F1 | 0.883 (B3, real+synth) | ⚠️ TBD (F22 fix at epoch 8) | ❌ Different paradigm (transition detection vs per-frame) |
| PSR POS | 0.797 (B3) | ⚠️ TBD | ❌ Same paradigm difference |
| **Ego-pose MAE** | **None — first baseline** | **8.92°** | ✅ Original contribution |
