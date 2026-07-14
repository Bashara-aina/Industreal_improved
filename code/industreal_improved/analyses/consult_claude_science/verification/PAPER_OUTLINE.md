# PAPER OUTLINE — Section-by-Section Readiness

**Working title (freeze Day 60):** "Multi-Task Industrial Assembly Perception: A Single-Backbone System for Detection, Activity, Procedure State, and Head Pose on IndustReal" (FINAL_PAPER_FRAMEWORK's ranked #1; #2 "Kendall-Capped MTL on IndustReal: 4 Tasks, 1 Backbone, 1 GPU" is the fallback if the method framing strengthens)
**Contingency title (if Q12 both-regimes framing wins):** "…: Pathologies and Remedies for Multi-Task Learning on IndustReal"
**Venue:** AAIML 2027 (IEEE Intl Conf on Advances in AI and Machine Learning, track "AI in Manufacturing" per A8) — deadline Oct 10, 2026 · **8 pages + references** (R4/V1 224; confirm per G8) · template `popw_aaiml2027.tex`
**Abstract:** a full draft exists in FINAL_PAPER_FRAMEWORK — reuse it, but fix its stale claims before lifting: the 312× gradient ratio (now 20,245× per A7/Q11) and the "Varifocal + WIoUv3" detection-loss description (flags are off in the live config).
**Forbidden claims (Item 54, binding):** no SOTA claim · no "novel MTL algorithm" · no generalizability claim · no deployment-ready claim · no fabricated numbers

**Status legend:** READY = writable today from verified facts · AWAITING = needs a scheduled run · PROGRESS = partially drafted

---

## §1 Introduction — **PROGRESS** (final pass after Day 30)
- **Key content:** industrial assembly perception needs 4 concurrent capabilities; deploying 4 single-task models costs 81M params / 8 fps (stacked baseline); one shared backbone: 46.47M params, 12–25 fps target.
- **Contributions list (defensible, per Items 51–52):**
  1. First multi-task system (4 heads, single backbone) on IndustReal.
  2. First head-pose baseline on IndustReal (with MediaPipe comparison).
  3. Characterization of three MTL pathologies on this benchmark with measured mechanisms (PSR dead-ReLU F1 0→0.7018; Kendall log-var collapse + cap; 20,245× gradient-norm imbalance) and remedies.
  4. ST-vs-MTL per-task accounting under a fixed budget, 3 seeds, bootstrap CIs.
- **Numbers needed:** final MTL/ST deltas (Day 30), efficiency table (Day 28).
- **Risk note:** write both framings of the last paragraph now (Q12); choose at Day 30.

## §2 Related Work — **READY except two paragraphs**
- **Ready now:** IndustReal + WACV 2024 (Schoonbeek; 24/75/11 taxonomy per Item 83); MTL surveys; loss weighting (Kendall, UW-SO, FAMO, DB-MTL arXiv:2308.12029 — corrected ID per Item 56); gradient surgery (PCGrad, Nash-MTL, CAGrad — cited as alternatives, not used); long-tail (LDAM-DRW, Balanced Softmax, cRT); head pose (6D rotation, geodesic losses, MediaPipe).
- **Awaiting:** Nardon differentiation ¶ (Day 4, G1) · FABRIC ATRE ¶ (Q49, wk 8–10).
- **Citations:** all 23 R3 citations verified real (Item 74); Q48 format audit wk 8.

## §3 Method — **READY** (all facts code-verified 2026-07-14; per-component numbers from R2's measured breakdown)
- **3.1 Architecture:** ConvNeXt-Tiny backbone (28.59M, ImageNet-1K) + standard FPN P3–P7 (4.48M); **46.47M total** (measured, Item 77). Temporal context: **TMA cell + FeatureBank (embed_dim 512, T=16)** — no video backbone (USE_VIDEOMAE=False). Heads: RetinaNet-style detection 5.31M (9 anchors × 24 classes × 5 levels); activity FeatureBank+TCN+2×ViT 0.69M (75-way clip); PSR causal head hidden=128, 3.08M (11 components, T=8 sequence); **two pose heads** — body pose 1.64M (17 COCO keypoints, *pseudo-labels from boxes* — limitation §6) and head pose 1.45M (real HL2 data, 6D rotation + huberised geodesic δ=30°, dropout 0.1); **PoseFiLM 0.84M + HeadPoseFiLM 0.40M** C5 modulators (must appear in the efficiency table — per the discrepancy report, omitting them is a known error mode).
- **3.2 MTL optimization:** Kendall uncertainty weighting with **per-task log-var clamps** (det (−4, 2), act (−0.5, 2), psr (−4, 0), pose (−4, 3), via `_clamp_kendall_log_vars`) + KENDALL_HP_PREC_CAP (pose precision ≤ det); KENDALL_STAGED_TRAINING=False (double-curriculum fix — worth one sentence); per-task LR multipliers (det 1.0×, act 3.0×, psr 0.5×, pose 0.3× — pending ablation #1 adoption); UW-SO alternative (ablated); PCGrad available on shared params; weight_decay=0 on log-vars.
- **3.3 Long-tail & imbalance:** LDAM-DRW with deferred re-weighting (epoch 50); PSR transition targets vs per-frame static labels (the Pathology-1 fix), **focal-BCE γ=0.5 — a deliberate deviation from the reference's γ=2.0, with the gradient-signal rationale (config comment) stated explicitly**; OHEM + asymmetric focal (γ⁺=0, γ⁻=1.5) + per-class alphas for detection.
- **3.4 Curriculum:** 10-stage RF1–RF10 progressive head unlocking + data scaling (stage_manager, not the legacy config.STAGED_TRAINING); OneCycleLR (not cosine); SWA disabled, EMA only; gate-based advancement with dynamic epochs; per-stage epochs and data fractions (RF1=20 ep/20%, RF2=30/50%, RF3=15/35%, RF4=20/50%, RF5–RF9=10 ep each scaling 50→90%, RF10=15/100%). Per-stage gate thresholds tighten progressively.
- **Training details:** batch 6 × grad-accum 8 = effective 48; bf16 only; grad clip 5.0; AdamW/Lion per config.
- **⚠️ Consistency check before writing:** FINAL_PAPER_FRAMEWORK §3.1.3 describes the detection loss as "Asymmetric Focal + Varifocal + WIoUv3," but the live config has `USE_VARIFOCAL=False` and `USE_WIOU=False` — **the paper must describe the flags actually on in the frozen config**, not the framework's aspirational table.
- **Figures:** architecture diagram (draw wk 2); gradient-norm bar chart (pose 3278 / act 13.8 / det 1.86 / psr 0.16 — data in hand); **Figure 2 Kendall log-var trajectories capped vs uncapped (data from ablation X1, Day 9–11)**.

## §4 Experiments — **AWAITING RESULTS** (all runs scheduled)
- **4.1 Setup — READY:** dataset facts (84 recordings, 27 subject-disjoint participants, 36/16/32 split, 207,266 frames, train stride 3 → 26,322 samples, native 1280×720@10FPS → 224×224 input); seeds [42,123,7] (+ escalation per Day-21 decision, deviation from Doc 223's N=5 acknowledged with per-sample bootstrap CIs per G9); protocols per metrics doc: activity = 16-frame clip top-1 majority vote; detection = mAP50 present-class (report n_present of 24); PSR = F1@±3-frame (authors' scorer per Q23); pose = geodesic MAE deg + position MAE mm; efficiency on RTX 3060 batch-1.
- **4.2 Main table (MTL vs ST, mean±std + bootstrap CI).** Two target sets exist in the archive — reconcile at Day 8 against real numbers:

| Task | Metric | Metrics-doc target | Framework target/stretch/fallback | WACV 2024 anchor (different protocol) |
|------|--------|-------------------|-----------------------------------|----------------------------------------|
| Activity | clip top-1 | 0.35–0.45 | 0.30 / 0.40 / 0.20 | 0.6525 (MViTv2-S, multi-modal) |
| Detection | mAP50-pc | 0.33–0.45 | 0.30 / 0.40 / 0.20 | 0.838 (YOLOv8m @1280px, ST) |
| PSR | F1@±3 | 0.50–0.62 | 0.15 / 0.25 / 0.05 | 0.883 (B3, transition paradigm) · STORM 0.506 |
| Head pose | MAE° | ≤15 (have ~9) | 7 / 5 / 10 | none (novel) |

  The WACV anchors go in the paper as context with the protocol caveats (resolution/modality/paradigm), NOT as head-to-head comparisons (forbidden-claims list).
- **4.3 Head-pose baseline table:** ours vs MediaPipe — MAE on covered frames + coverage % + occlusion breakdown (Day 2; framing per G3).
- **4.4 Ablations (each 50-ep vs baseline epoch-50 reference):** **uncapped vs capped Kendall (X1 — Table 5 row 1)**; loss weighting (Kendall vs UW-SO[+log1p]); BiFPN vs FPN; per-task LR on/off (implicit in ablation #1); gated extras as run (TSBN/ASL/MetaBalance/MViTv2-S/cRT/OHEM).
- **4.5 Efficiency:** params / GFLOPs / FPS vs 4-model stack (81M, 8 fps) — Day 28; include FiLM modules in the param accounting.
- **4.6 Error analysis:** activity confusion matrix (Day 9); PSR per-component positive rates + per-component F1 (Day 3 + Day 30); constant-prediction floor for PSR; class-0 = `take_short_brace` semantics stated (Q37).

## §5 Discussion — **PROGRESS** (both framings pre-written by Day 20)
- **Framing A (default):** MTL reaches parity-or-better on k of 4 tasks at 43% of stacked params; where it loses, the measured pathology explains why.
- **Framing B (contingency, if ST also weak):** IndustReal is hard in both regimes; contribution = pathology characterization + remedies + honest per-task accounting.
- **Must include:** test-vs-val gap (Q14, Day 56–60); seed variance discussion (Q13); why per-frame PSR metrics mislead (all-ones degenerate solution — from metrics doc).

## §6 Limitations — **READY to draft wk 7**
- 224px input ceiling for detection (V1 doc 212 — why anchor-free deferred); body pose = pseudo-keypoints from boxes (Q38); single dataset (no generalizability claim, per forbidden list); 3 seeds (justify via G9 protocol); MediaPipe comparison is RGB-only vs multi-modal alternatives excluded.

## §7 Conclusion + Future Work — **READY to draft wk 7**
- Future work inventory comes free from the NO-GO/DEFER lists: Nash-MTL/CAGrad/RotoGrad (direction-space surgery), ConsMTL (bi-level), anchor-free + higher-res detection, per-task augmentation, distillation from ST teachers (wiring exists), TSBN/TAL as detection remedies.

## Reproducibility appendix — **READY**
- Full config table (all flags with defaults: USE_KENDALL=True, USE_LDAM_DRW=True/epoch 50, USE_PSR_TRANSITION=True/σ=3, USE_GEO_HEAD_POSE=True, USE_BIFPN per ablation, DET_OHEM=True; implemented-not-ablated: FAMO, MetaBalance, RotoGrad, TAL, Varifocal, WIoU, ASL, BalancedSoftmax).
- Seeds, hardware (RTX 3060 12GB + RTX 5060 Ti 16GB), GPU-hours per experiment (from COMPUTE_SCHEDULE ledger — reviewers increasingly value this).

---

## Figure/table manifest

| Asset | Type | Data source | Due |
|-------|------|------------|-----|
| F1 Architecture diagram | fig | §3 facts | wk 2 |
| F2 Kendall log-var trajectories, capped vs uncapped | fig | X1 ablation (Day 9–11) | Day 12 |
| F3 Gradient-norm imbalance | fig | A7 measurements (in hand) | wk 2 |
| F4 Training curves w/ pathology annotations | fig | baseline run logs | Day 9 |
| F5 Activity confusion matrix | fig | Day 9 run | Day 10 |
| F6 Qualitative: det boxes + pose arrows | fig | seed-42 checkpoint | Day 22–24 |
| (opt) F7 MTL/ST transfer map or efficiency radar | fig | framework Figure 3/4 designs, Day 30 data | wk 5, if page budget allows |
| T1 Main MTL vs ST (3 seeds, CIs) | table | Day 30 aggregate | Day 31 |
| T2 Pose vs MediaPipe | table | Day 2 | Day 9 |
| T3 Ablations | table | Days 8–14 | Day 15 |
| T4 Efficiency | table | Day 28 | Day 29 |
| T5 PSR per-component | table | Day 3 + Day 30 | Day 31 |

## Writing schedule (from 30_DAY_EXECUTION_PLAN)
§3+§4.1 → wk 1 · §2 → wk 2 · §1 + both §5 framings → wk 3 · §4 fill → wk 5 · full draft v1 → wk 6 · review+§6/§7 → wk 7 · test numbers + draft v2 → wk 8–9 · audits (Q48/Q49) + external read → wk 10–11 · final → Oct 8 submit.

**End of PAPER_OUTLINE.md**
