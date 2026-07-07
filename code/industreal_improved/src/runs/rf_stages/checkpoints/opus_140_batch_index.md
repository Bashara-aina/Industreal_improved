# Opus 140/141 Session Index — 2026-07-06/07

**Freeze checkpoint:** `best.pth` (epoch 18, sha256: `59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8`)
**Paper freeze:** Jul 20
**Total commits this session:** 47 (since 2026-07-06 JST)

---

## 1. Detection

| Hash | Description |
|------|-------------|
| `6fdb88981` | Error-state FPR analysis — class 24 has 0 GT instances; model FPR=0.0% vs WACV's 65% (trained on real errors) |
| `b3591481b` | WACV mAP convention check + zero-GT count verification — 6 zero-GT classes, not 9 as earlier subsample suggested |
| `15dd1c07d` | Single-task ConvNeXt-Tiny detection training launch script (Opus 140 Q4) |
| `86ffb3436` | D1 weights — fail hard on IndustReal download failure instead of silent COCO fallback (Opus C-2) |
| `bc7a03761` | Comprehensive AAIML strategy files + eval pipeline + bug fixes |

## 2. Head Pose

| Hash | Description |
|------|-------------|
| `bff38b790` | Fix `head_pose_diag.py` up-vector index bug — slice [6:9] not [3:6] (3.5-month bug, same as eval scripts) |
| `4a3487b93` | Head pose Kalman smoothing results — single-frame vs RTS-smoothed (Opus HP-6) |
| `9caba66c2` | Per-recording forward MAE + FiLM stats + pose GT variance (Opus 140 Day 1) |
| `911fb29c7` | Pose between/within recording variance decomposition (Opus 141 Q25) |
| `3dedebdf2` | Pose outlier analysis + GT noise floor — outlier recording is model failure, not GT artifact (Opus 141 Q14, Q21, Q29) |

## 3. Activity

| Hash | Description |
|------|-------------|
| `dcd32e3ef` | Fix activity linear probe NaN bug — filter -1 labels, gradient clipping, feature caching (36 min vs ~10 hr) |
| `4f9909a01` | Activity confusion matrix — verb-antonym evidence (takeX↔putX = 1.3% of errors, temporally ambiguous) (Opus ACT-4) |
| `7001107de` | Activity temporal probe — fixed ConvNeXtBackbone, temporal pooling over 16-frame clips (Opus ACT-1) |

## 4. PSR (Procedure State Recognition)

| Hash | Description |
|------|-------------|
| `d0c186827` | PSR LOO-CV confirms per-comp threshold gain is real — +0.0358 ± 0.0216 held-out improvement |
| `944add8c0` | PSR per-comp optimal F1 on full 38k val set — macro F1 = 0.7499 (Opus 140 Q2) |
| `6bc786efc` | PSR null-delta table — learned signal quantified for low-prevalence components (Opus PSR-6) |
| `fc80c97d3` | Null-model POS proves fill-forward artifact — all-zeros POS=0.9995, copy-prev 0.9984, vs ours 0.9988 (Opus Q3, PSR-1) |
| `1fb744f03` | Retract misleading SOTA labels in SOTA_STATUS.md per 140 §-1 |
| `2989801dc` | PSR head repair (LeakyReLU + bias=0.0 + Xavier) with PSR_HEAD_REPAIR env toggle |
| `96b144e51` | PSRHead activation diagnostic + real-repair design (Opus 140 Q8, -1d) |
| `e618d929a` | Fix PSRHead output heads — LeakyReLU + small-normal init + zero bias (Opus 140 §-1d) |
| `6defe1f5f` | Fix PSRHead init index — Sequential[3]=Linear, [2]=Dropout (no weight) |
| `94c1b5e71` | PSR LOO-CV stratified by train/val membership (Opus 141 Q20) |
| `a63c21c02` | 24→11 PSR mapping verified — _build_psr_mask() produces correct PSR_MASK matrix (Opus 141 Q34) |
| `9cf32fe2b` | Bootstrap CI for headline numbers — head pose, PSR F1 with 95% CIs across 16 recordings (Opus 141 Q1.10) |

## 5. D4 (YOLOv8m → MonotonicDecoder)

| Hash | Description |
|------|-------------|
| `dfbb3d6f6` | D4 threshold re-tune sweep — F1=0.000 (default Q48) → 0.347 (hi=0.3, lo=0.1, min=2) (Opus Q2, PSR-4) |
| `64aaeaa20` | D4+D1R decisive test — F1=0.636 with D1R fine-tuned YOLOv8m + retuned thresholds (Opus 140 Q10) |

## 6. Disclosures & Verification

| Hash | Description |
|------|-------------|
| `a7de2c140` | Training loss index verification — refutes 137 debate worst-case, loss uses correct [6:9] |
| `02a94937e` | Commit evidence artifacts + eval logs (Opus 132 §7) |
| `0da92b238` | Commit 3 missing evidence directories + D1R results.csv is already tracked |

## 7. Opus Q&A / Documentation

| Hash | Description |
|------|-------------|
| `2bc0bbf71` | File 139 — Opus overview prompt v2 synthesizing 132-138 |
| `f08bb2aed` | File 140 — Opus answers v2 to the 139 overview |
| `b863c6b49` | File 141 — complete per-question verdicts for files 134-138 |
| `766a3099d` | File 132 — Opus answers, audit of 131 overview, top-10 verdicts |
| `623c63fb2` | File 133 — Opus complete answers, all 66 questions, all 30 debates |
| `029301f05` | File 134 — 50 deep detection questions + debate |
| `fff2e736d` | File 135 — 50 deep PSR questions + debate |
| `bf00e9613` | File 137 — 50 deep head pose questions + debate |
| `2a9fb2ab4` | File 138 — 50 SOTA integration questions + beat plan |
| `b143ec635` | File 137 debate — adversarial review of head pose questions |
| `87b72e1da` | File 136 debate — adversarial review of activity questions |
| `1466fc53a` | File 134 debate — adversarial review of detection questions |
| `a0e76572f` | File 135 debate — adversarial review of PSR questions |
| `0c6c881be` | File 138 debate — adversarial review of integration plan |
| `cded04fb4` | Merge pull request #26 (opus-overview-v2 branch) |
| `6d8bc67bd` | Merge pull request #25 (aaiml-2027-opus-overview) |
| `683919de8` | Merge pull request #24 (aaiml-2027-opus-overview) |

---

## Summary by Area

| Area | Commits |
|------|---------|
| Detection | 5 |
| Head Pose | 5 |
| Activity | 3 |
| PSR | 12 |
| D4 | 2 |
| Disclosures | 3 |
| Opus Q&A/Docs | 17 |
| **Total** | **47** |
