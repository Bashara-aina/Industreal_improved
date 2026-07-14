# LITERATURE GAPS — Answers Still Needed from Claude Science

**Purpose:** Questions whose resolution requires literature search rather than code or training. Each entry gives ready-to-run queries for the paper-search MCP, what we already know, the evidence needed, and which checklist decision the answer moves.

> **Note (2026-07-14):** the paper-search MCP was not available in the session that produced this document; queries below are formulated to be executed verbatim in a Claude Science session.

---

## G1. Novelty re-verification vs Nardon and successors (→ Q46, Q52/Item 52)

- **Queries:**
  - `"IndustReal" multi-task learning`
  - `"IndustReal" dataset benchmark 2026`
  - `industrial assembly multi-task detection "procedure step" OR "assembly state" 2025..2026`
  - `arXiv:2506.15285 citing articles` (papers citing Nardon)
- **What we know:** R3 found 0 papers doing 4-task MTL on IndustReal; Nardon (arXiv:2506.15285) is single-task detection + state tracking on different data, threat assessed LOW (A19). 11 adjacent 2025–2026 papers catalogued.
- **Evidence needed:** any preprint after 2026-06 that trains ≥2 heads on IndustReal or claims a head-pose baseline on it.
- **Decision impact:** if found → rewrite novelty claims (§1, §2) from "first" to "first to jointly…" differentiation; does NOT change the experimental plan.
- **When:** Day 4 (initial) and **Day 80 (Oct 1) refresh — mandatory pre-submission**.

## G2. UW-SO temperature and stability at extreme loss-scale ratios (→ Q5, Q17 gate)

- **Queries:**
  - `Kirchdorfer uncertainty weighting softmax multi-task IJCV 2025`
  - `UW-SO temperature sensitivity multi-task loss weighting`
  - `softmax loss weighting scale invariance multi-task "log transform"`
- **What we know:** UW-SO weights = softmax(−L/T); our raw losses span ~4 orders of magnitude (pose grad 3278 vs psr 0.16), which may saturate the softmax at T=1.
- **Evidence needed:** does Kirchdorfer (or follow-ups) normalize/log-transform losses before softmax? Recommended T for heterogeneous task mixes (regression + detection + multilabel)?
- **Decision impact:** determines whether Q17 (DB-MTL log1p) should be fused into the UW-SO wiring *from the start* instead of gated — a 30-minute code difference that could save a wasted 25 GPU-h ablation.
- **When:** before Day 6 (ablation #1 launch). **Highest-urgency gap in this file.**

## G3. MediaPipe head-pose accuracy under occlusion/industrial conditions (→ Q4 framing)

- **Queries:**
  - `MediaPipe head pose estimation accuracy occlusion evaluation`
  - `head pose estimation industrial safety helmet occlusion benchmark`
  - `face mesh failure rate egocentric OR top-down camera head pose`
- **What we know:** MediaPipe ≈5° MAE on frontal controlled data; IndustReal has top-down-ish views, helmets/occlusion; our MAE ≈9°.
- **Evidence needed:** published MediaPipe degradation numbers under non-frontal/occluded conditions, and precedent for reporting *coverage* (fraction of frames with any estimate) alongside MAE.
- **Decision impact:** shapes the Q4 comparison table design (MAE-on-covered-frames + coverage% columns) and the rebuttal if MediaPipe wins raw MAE.
- **When:** by Day 9 (before the pose table is drafted).

## G4. Event-detection F1 at <1% positive rate — nearest published protocols (→ Q10, Q15, paper §4)

- **Queries:**
  - `rare event detection F1 tolerance window video temporal action`
  - `procedure step recognition evaluation "F1" tolerance frames assembly`
  - `extreme class imbalance "positive rate" below 1% multilabel temporal detection`
  - `asymmetric loss ASL rare positive multilabel video`
- **What we know:** R3 §3.2 — no published solution at <0.5% positive rate (Item 76); STORM reports PSR F1 0.506 as nearest anchor; ASL is our gated remedy.
- **Evidence needed:** (a) any 2024–2026 paper handling <1% event rate we could cite as method precedent for ASL-on-PSR; (b) confirmation that ±3-frame tolerance F1 is the accepted IndustReal PSR protocol (authors' scorer semantics).
- **Decision impact:** Q15 gate confidence; also protects against a reviewer claiming our tolerance protocol is non-standard.
- **When:** by Day 12 (slot #3 decision) for (a); Day 1 local `ls` (Q23) partially answers (b).

## G5. cRT / decoupled retraining on video activity (not image) classification (→ Q8 gate)

- **Queries:**
  - `decoupled representation classifier long-tail video action recognition`
  - `classifier retraining cRT long-tailed activity recognition video 2023..2026`
  - `LDAM-DRW versus decoupled training comparison long-tail`
- **What we know:** Kang ICLR 2020 established cRT on images; LDAM-DRW already active in our pipeline (DRW at epoch 50); Q8 fires only if activity top-1 < 0.35.
- **Evidence needed:** does cRT stack with (rather than substitute for) margin-based losses on video long-tail? Expected gain magnitude on ~75-class, heavily-skewed video data.
- **Decision impact:** if literature says cRT+LDAM conflict, the Q8 trigger should *swap* (disable DRW during retrain) rather than stack — changes ~10 lines in `decoupled_act_retrain.py`.
- **When:** by Day 8 (gate day).

## G6. TSBN evidence beyond NYUv2 (→ Q7 gate)

- **Queries:**
  - `task-specific batch normalization multi-task detection segmentation results`
  - `"task-specific" OR "task-conditional" normalization multi-task learning 2023..2026 detection mAP`
- **What we know:** conflicting — original claim "recovers 75% of det mAP gap" vs FINAL §2.5 correction: on NYUv2 TSBN *hurts* segmentation while helping depth.
- **Evidence needed:** any result where TSBN helps a *detection* head specifically in a shared-backbone MTL (our exact configuration).
- **Decision impact:** if no detection-positive evidence exists, downgrade Q7 from DEFER to NO-GO even if the mAP gate fires (spend the 25 GPU-h on TAL or threshold work instead).
- **When:** by Day 8.

## G7. MViTv2-S vs ConvNeXt-Tiny on frame-based (non-clip) multi-task inference (→ Q39 gate)

- **Queries:**
  - `MViTv2 image classification transfer detection comparison ConvNeXt`
  - `video transformer backbone multi-task dense prediction efficiency comparison`
- **What we know:** V1 doc 214 projects +10–15% activity from MViTv2-S; our pipeline for it exists (`scripts/train_mtl_mvit.py`); cost 50 GPU-h; VRAM risk on 16GB.
- **Evidence needed:** published MViTv2-S vs ConvNeXt-T deltas on *detection and regression* heads (not just classification) — does the activity gain come at detection/pose cost?
- **Decision impact:** whether a triggered Q39 replaces the final config (risky, late) or is reported as an ablation only (recommended).
- **When:** by Day 12 (slot decision).

## G8. AAIML 2027 formatting, page limit, review criteria (→ Phase 4)

- **Queries (web, not paper-search):**
  - `AAIML 2027 IEEE call for papers page limit format`
  - `AAIML 2026 accepted papers proceedings` (calibrate typical rigor/length)
- **What we know:** deadline Oct 10, 2026 verified (A8); template in `popw_aaiml2027.tex`.
- **Evidence needed:** exact page limit incl./excl. references; supplementary policy; double-blind or not (affects repo/artifact anonymization).
- **Decision impact:** Phase-4 writing budget and whether ablation tables move to supplementary.
- **When:** Week 6 (before draft v1 is size-committed).

## G9. Bootstrap CI protocol for 3-seed reporting (→ Q2, Q13)

- **Queries:**
  - `bootstrap confidence interval few seeds deep learning reporting best practice`
  - `"mean ± std" versus confidence interval 3 seeds machine learning evaluation`
- **What we know:** metrics protocol demands seeds [42,123,7] + bootstrap CIs; 3 seeds is thin for CIs over seeds (bootstrap should be over *test samples* per seed, then aggregated).
- **Evidence needed:** a citable protocol (e.g., Bouthillier et al.) for small-seed-count reporting to preempt reviewer statistics objections.
- **Decision impact:** exact wording + CI method in §4; zero compute.
- **When:** Week 5.

---

## Priority order for the next Claude Science session

| # | Gap | Deadline | Blocks |
|---|-----|----------|--------|
| 1 | G2 UW-SO temperature/log-transform | Day 6 | ablation #1 design |
| 2 | G5 cRT×LDAM interaction | Day 8 | Q8 trigger shape |
| 3 | G6 TSBN detection evidence | Day 8 | Q7 gate validity |
| 4 | G3 MediaPipe occlusion | Day 9 | pose table framing |
| 5 | G4 <1% event-rate precedent | Day 12 | Q15 slot decision |
| 6 | G7 MViT trade-offs | Day 12 | Q39 slot decision |
| 7 | G1 novelty refresh | Day 80 | submission |
| 8 | G8 AAIML format | Week 6 | draft v1 |
| 9 | G9 CI protocol | Week 5 | §4 wording |

**End of LITERATURE_GAPS.md**
