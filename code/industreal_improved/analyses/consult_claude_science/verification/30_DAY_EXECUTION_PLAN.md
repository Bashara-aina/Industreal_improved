# 30-DAY EXECUTION PLAN — Jul 14 → Aug 12, plus weekly plan to Oct 10

**Companion:** `AAIML_SUBMISSION_CHECKLIST.md` (decisions + gates), `COMPUTE_SCHEDULE.md` (GPU ledger), `RISK_REGISTER.md` (fallback triggers)
**Convention:** GPU 0 = RTX 3060 12GB (116 GPU-h budget) · GPU 1 = RTX 5060 Ti 16GB (365 GPU-h budget) · thermal cap ≈ 16 combined GPU-h/day
**Seeds:** 42, 123, 7 (per project metrics protocol)

---

## WEEK 1 (Jul 14–20): PHASE 0/1 — FOUNDATION
Everything CRITICAL starts now; all zero-GPU wiring lands while GPUs are busy.

### Day 1 — Mon Jul 14
- **FIRST (before any launch, ~1 h): pre-launch config review** — see AAIML_SUBMISSION_CHECKLIST "Day-1 pre-launch config review". Decide **`FREEZE_BACKBONE`** explicitly (config.py:199 currently `True` = linear probe; frozen-probe activity ceiling is 0.2169 vs 0.35 target — default decision: `False`/fine-tune for both ST and MTL runs unless RF1-stage unfreezing is confirmed). Set determinism env per Doc 223 (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, cudnn deterministic). Confirm fixed split file. Same backbone mode for ST and MTL or the comparison is invalid.
- **GPU 0:** Smoke ST scripts: `bash scripts/launch_st_baselines.sh --dry-run`, then 1-epoch run per head (~2 GPU-h total). If clean → launch **ST pose × 3 seeds** (10.5 GPU-h, runs into Day 2). Edit seed list to `42 123 7` first if hardcoded to 5.
- **GPU 1:** Launch **main MTL baseline** — reviewed config, seed 42, 100 epochs (~50 GPU-h at the conservative estimate, runs through Day 5). This one run answers Q3 (det mAP), Q9 (activity top-1), Q10 (PSR F1) and its epoch-50 checkpoint becomes the ablation reference.
- **Evening: throughput measurement** — read min/epoch at epochs 2–5 and rewrite the COMPUTE_SCHEDULE ledger. The archive prices a 100-ep MTL run at 10 h (Doc 226) / 50 h (V2 framework) / 96 h (Doc 222) — the measured number governs the seed-escalation decision (≤20 h/run → 5 seeds per Doc 223) and every later slot.
- **Code (while training runs):** Wire UW-SO (Q5, 1.5 h — config flag + branch in `src/training/losses.py`). Wire per-task LR (Q6, 1 h — `PSR_LR_MULTIPLIER=0.5`, `HEAD_POSE_LR_MULTIPLIER=0.3` in config + `train.py` param groups ~3879–3899, both Lion and AdamW branches).
- **Local machine (5 min):** Q23 — `ls /media/newadmin/master/POPW/datasets/industreal/`, confirm authors' PSR scorer present; clone reference repo if not.
- **Expect by EOD:** both GPUs saturated; UW-SO + per-task LR committed behind flags (off); ledger re-priced from measured throughput.

### Day 2 — Tue Jul 15
- **GPU 0:** ST pose finishes (~noon). Run **MediaPipe pose baseline** (Q4, ~2 GPU-h) against ST-pose seed-42 checkpoint on identical val protocol. Then launch **ST activity × 3 seeds** (15 GPU-h, into Day 3).
- **GPU 1:** Main baseline continues (monitor: LIVENESS logs, Kendall grad logs every 500 steps).
- **Code:** EMA warmup (Q24, 0.5 h — `EMA_START_EPOCH=5`, guard train.py:1518 + ~2040). Wire Balanced Softmax flag (Q16, 1 h, off). Wire ASL flag (Q15, 1 h, off). 6D round-trip unit test (Q40 guard, 0.5 h).
- **Expect:** MediaPipe MAE number in hand → first paper table row. Gate (per V2 plan Day-0): if MediaPipe MAE < 4°, the pose contribution needs reframing now, not at writing time; if merely < ours, activate the coverage + occlusion breakdown framing (RISK_REGISTER R6).

### Day 3 — Wed Jul 16
- **GPU 0:** ST activity continues.
- **GPU 1:** Main baseline continues. Pull epoch-20 partial eval — early warning for R1 (det mAP ≈ 0 trigger at epoch 30).
- **Code/CPU:** Q36 per-component PSR positive-rate analysis (1 h, pandas). Q37 activity class-0 semantics — read dataset loader ID map + annotation docs, write the verdict into paper notes (1 h). Q38 body-pose provenance paragraph (0.5 h).
- **Expect:** the two data conflicts (Q36/Q37) resolved on paper-quality evidence.

### Day 4 — Thu Jul 17
- **GPU 0:** ST activity finishes (~evening). Launch **ST PSR × 3 seeds** (15 GPU-h, into Day 6).
- **GPU 1:** Main baseline continues.
- **Code:** **Q46 MANDATORY: read Nardon arXiv:2506.15285 itself** — the archive contradicts itself on what this paper is (A19: LOW threat, detection+state tracking vs A9: MODERATE threat, 6-DoF head-pose estimator); resolve from the primary source, write the differentiation paragraph, and adjust the "first head-pose baseline" wording if needed (3 h). Smoke-test the candidate config end-to-end: 1-epoch run with `USE_UW_SO=1` + per-task LR + EMA warmup on GPU 1's idle margin or CPU-debug (loss decreases on all 4 tasks, weights logged).
- **Expect:** candidate config proven runnable before it's needed on Day 8; Nardon identity settled.

### Day 5 — Fri Jul 18
- **GPU 0:** ST PSR continues.
- **GPU 1:** **Main baseline finishes (~50 GPU-h consumed).** Run full eval suite on best + epoch-50 checkpoints: `eval_activity_75class.py` (clip top-1), `eval_psr_transition_f1.py` (±3-frame), det mAP50-pc dual protocol, pose MAE. Run PSR constant-prediction diagnostic (floor).
- **Expect by EOD:** **the numbers that drive every gate: det mAP50-pc, activity clip top-1, PSR event-F1, pose MAE.**

### Day 6 — Sat Jul 19
- **GPU 0:** ST PSR finishes. Launch **ST detection × 3 seeds** (21 GPU-h, into Day 8).
- **GPU 1:** Launch **Ablation slot #1: candidate config** (UW-SO + per-task LR + EMA warmup), seed 42, 50 epochs (~25 GPU-h, into Day 8).
- **Analysis:** Draft the ST-vs-MTL comparison table skeleton with pose/activity/PSR ST numbers as they land.

### Day 7 — Sun Jul 20
- **GPU 0:** ST detection continues.
- **GPU 1:** Ablation #1 continues.
- **Paper:** Start §3 (Method) and §4.1 (Setup) — fully writable now (architecture facts verified: ConvNeXt-Tiny 28.59M, 46.47M total, heads, losses, curriculum).

**Week-1 exit criteria:** main-baseline metrics known; ST pose/act/psr done ×3 seeds; MediaPipe number known; all Phase-1 wiring merged; ~63 GPU-h burned on GPU 0, ~75 on GPU 1.

---

## WEEK 2 (Jul 21–27): PHASE 2 — GATE READS + ABLATIONS

### Day 8 — Mon Jul 21 ⭐ **GATE DAY**
- **GPU 0:** ST detection finishes (~evening). **GPU 0 training budget now ~90% consumed — GPU 0 becomes the eval/figures device.**
- **GPU 1:** Ablation #1 (candidate config) finishes. Compare vs baseline epoch-50 reference.
- **Read gates (from AAIML_SUBMISSION_CHECKLIST):**
  - det mAP50-pc < 0.33? → trigger Q7 (TSBN, 6 h build) else skip
  - activity top-1 < 0.35? → trigger Q8 (cRT retrain, cheap) and provisionally Q39 (MViT)
  - PSR F1 < 0.50? → queue Q15 (ASL) for slot #3
  - UW-SO Δ vs Kendall: adopt / reject / (inconclusive → Q43 FAMO tiebreaker)
  - pose/psr grad ratio still >1000×? → queue Q44 (MetaBalance) for slot #3
- **Decide slot #3 and #4 occupants tonight.** Slot #2 is pre-committed to the **uncapped-Kendall control (X1)** — paper-critical (Table 5 row 1 + Figure 2), ungated. Default for the rest (no gates fire): #3 = BiFPN (Q41), #4 = OHEM-off (Q28) or none.

### Day 9 — Tue Jul 22
- **GPU 0:** Q22 confusion matrix on best activity checkpoint (1 GPU-h). ST-baseline bootstrap CIs + `metrics.json` aggregation script.
- **GPU 1:** Launch **Ablation slot #2: uncapped Kendall** — `bash scripts/launch_uncapped_kendall.sh` adapted to 50 epochs, seed 42, ~25 GPU-h (into Day 11). Log-var trajectories every 500 steps feed Figure 2 directly.
- **Paper:** §4.2 ST baseline table with real numbers.

### Day 10 — Wed Jul 23
- **GPU 0:** idle/eval margin (thermal recovery; budget preservation).
- **GPU 1:** Slot #2 continues.
- **Code:** if Q8 triggered: adapt `decoupled_act_retrain.py` (4 h) — it runs on GPU 0's remaining margin (5 GPU-h) Day 11. If a detection gate fired: TSBN build (6 person-h) today so it can occupy slot #3.
- **Cloud (X3):** provision RunPod/Lambda account + 1-epoch test run (~$5) so the R8 fallback is exercised.

### Day 11 — Thu Jul 24
- **GPU 0:** (if triggered) cRT activity retrain, 5 GPU-h.
- **GPU 1:** Slot #2 finishes → **launch slot #3: BiFPN** (`USE_BIFPN=1`, 50 ep, ~25 GPU-h, into Day 13) — or TSBN if the detection gate fired.
- **Analysis:** uncapped-vs-capped comparison written down (Figure 2 data secured).

### Day 12 — Fri Jul 25
- **GPU 1:** Slot #3 continues.
- **Paper:** §2 Related Work full draft (Nardon paragraph from Day 4, FABRIC ATRE placeholder for Day 62).

### Day 13–14 — Sat–Sun Jul 26–27
- **GPU 1:** Slot #3 finishes Day 13 → decision: BiFPN in/out of final config (adopt if det mAP50-pc +≥1.0 with no other task regressing >0.5). **Slot #4 (gated)** launches if queued: ASL (PSR weak) / MetaBalance (grads unbalanced) / OHEM-off (all healthy) / **none** (bank the reserve). MViTv2-S only per Q39's raised bar (legacy-code revival cost — see checklist).
- **GPU 0:** figures pipeline: training curves (`plot_training_curves.py`), gradient-norm figure (Pathology 3 with 20,245× annotation), PSR transition-target illustration.
- **Analysis:** full ablation table assembled; every number that will enter the final config now exists. Doc 222 compute-savers apply to slot #4 (25-epoch weighting sweeps; epoch-10 neck ranking).

**Week-2 exit criteria:** all gates resolved; ablation table complete; final config candidates reduced to one.

---

## WEEK 3 (Jul 28–Aug 3): PHASE 2 CLOSE — FREEZE

### Day 15–17 — Mon–Wed Jul 28–30
- **GPU 1:** contingency window — rerun any ablation that crashed/ambiguous; else start final-config run early (becomes seed-42 of Phase 3 — saves 50 GPU-h of Phase 3 if config == an ablation config, in which case that ablation run IS seed 42 extended to 100 epochs: resume from its epoch-50 checkpoint, +25 GPU-h only).
- **GPU 0:** eval-protocol dry run end-to-end: one command produces every paper table from a checkpoint dir (build `generate_paper_table.py` glue, script exists).

### Day 18–20 — Thu–Sat Jul 31–Aug 2
- **GPU 1:** final-config seed 42 completes 100 epochs.
- **Paper:** §1 Intro draft; both §5 framings pre-written (MTL-competitive vs pathology-characterization, per Q12).

### Day 21 — Sun Aug 3 ⭐ **ARCHITECTURE FREEZE**
- Final config locked in `src/config.py` + tag `git tag aaiml-freeze`. After today, config changes only via RISK_REGISTER triggers.
- Phase-3 GPU ledger reviewed (see COMPUTE_SCHEDULE): GPU 1 must have ≥ 125 h remaining for seeds 123 + 7 + margin. If not, drop to 2 additional seeds is NOT allowed — instead cut slot-4-style extras; 3 seeds is the floor.
- **Seed-escalation decision (Doc 223 compliance):** if the measured run cost is ≤20 h (Day-1 measurement), schedule seeds 4–5 (e.g., 1000, 2026) into Weeks 4–5 to meet the protocol's N=5 for main experiments; otherwise document the 3-seed deviation + per-sample bootstrap CIs (G9) in §4.1.

---

## WEEK 4 (Aug 4–10): PHASE 3 — MULTI-SEED

### Day 22–24 — Mon–Wed Aug 4–6
- **GPU 1:** final config **seed 123**, 100 epochs (~50 GPU-h).
- **GPU 0:** per-seed evals of seed 42; qualitative figures (detection visualizations, pose arrows on frames).

### Day 25–27 — Thu–Sat Aug 7–9
- **GPU 1:** seed 123 finishes → **seed 7** launches (~50 GPU-h, into Day 30).
- **Paper:** §4 Results skeleton filled with seed-42 + ST numbers; CIs marked TBD.

### Day 28 — Sun Aug 10
- **GPU 0:** efficiency measurements (`scripts/measure_efficiency.py`: params, GFLOPs, FPS batch-1 on the 3060 — the metrics doc specifies 3060 for efficiency reporting).
- **Buffer/rest day.**

---

## Day 29–30 — Mon–Tue Aug 11–12
- **GPU 1:** seed 7 completes.
- **Analysis:** 3-seed aggregation, bootstrap CIs, mean±std for every table. **The paper's full quantitative core exists by Day 30.**
- If Q42 gate fired (MTL < ST on ≥2 tasks): decide now whether reserve (~75–90 h) funds teacher-cache + distilled run (55 h) — hard cutoff for starting it is Day 33.

---

## WEEKLY PLAN — Day 31 → submission (Aug 13 → Oct 10)

| Week | Dates | GPU work | Paper work |
|------|-------|----------|-----------|
| 5 | Aug 13–19 | Gated extras from reserve (distill run / OHEM / seeds 4–5 if escalation fired); else idle | Full Results §4 with CIs; ablation table final; all figures v1 |
| 6 | Aug 20–26 | Idle (reserve held for emergencies) | Complete draft v1 end-to-end; Q11 stale-number sweep (grep 312/140x) |
| 7 | Aug 27–Sep 2 | — | Internal review pass 1; limitations §; reproducibility appendix (flags: FAMO/MetaBalance/RotoGrad/TAL implemented-not-ablated). **X2 code-release track starts: repo cleanup, README, `pip freeze > requirements_frozen.txt`, reproduce script** |
| 8 | Sep 3–9 | **Day 56–60: single test-split eval (Q14, 3 GPU-h)** on frozen final-seed checkpoints — ONCE, never repeated | Test numbers into paper; val-vs-test gap discussion. **X2: checkpoint archival (Zenodo DOI) + weights upload** (respect AAIML anonymity policy per G8) |
| 9 | Sep 10–16 | — | **Day 60 (Sep 11): title freeze (Q47).** Draft v2; polish figures |
| 10 | Sep 17–23 | — | Day 62-equivalents done in wk 8–10: Q48 citation audit, Q49 FABRIC ATRE ¶; external/advisor read |
| 11 | Sep 24–30 | Emergency re-runs only | Address advisor comments; camera-ready formatting vs AAIML template |
| 12 | Oct 1–7 | — | **Q46 novelty re-search (Day 80, Oct 1).** Final proofread; submission dry-run on portal |
| 13 | Oct 8–10 | — | **Submit Oct 8 (2-day buffer). Hard deadline Oct 10.** |

---

## Standing rules

1. **Never leave GPU 1 idle in Weeks 1–4** — it holds 76% of the budget and every day of idleness is ~10 GPU-h of ablation capacity lost.
2. **GPU 0 stops training after Day 11** (budget ~95% spent) — it becomes eval/figures only.
3. **Every run writes `metrics.json` + tensorboard into `runs/<name>/seed_<n>/`** matching the ST launcher convention, so aggregation is uniform.
4. **Gates are read once, on gate day, from written-down numbers** — no re-litigating a gate because a later checkpoint looked better.
5. **Any crashed run gets one retry; second crash → escalate to RISK_REGISTER fallback,** don't silently burn budget.
6. Commit + push after every wiring change; `npx gitnexus analyze` after commits per CLAUDE.md.

**End of 30_DAY_EXECUTION_PLAN.md**
