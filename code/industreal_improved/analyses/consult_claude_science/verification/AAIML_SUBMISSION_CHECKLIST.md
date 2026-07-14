# AAIML SUBMISSION CHECKLIST — The Definitive Gate

**Date:** 2026-07-14
**Deadline:** AAIML 2027 submission — Oct 10, 2026 (88 days)
**Inputs:** `MASTER_VERIFICATION.md` (85 items), `UNANSWERED_QUESTIONS.md` (50 questions)
**Codebase re-verified:** 2026-07-14 against `code/industreal_improved/` (this session; all file:line refs checked)
**Budget:** GPU 0 (RTX 3060 12GB) ~116 GPU-h · GPU 1 (RTX 5060 Ti 16GB) ~365 GPU-h · combined realistic throughput ~16 GPU-h/day

## Decision Summary

| Decision | Count | Questions |
|----------|-------|-----------|
| **GO** | 24 | Q1–Q6, Q9–Q14, Q19, Q22–Q24, Q36–Q38, Q41, Q46–Q50 |
| **DEFER** (gated) | 10 | Q7, Q8, Q15, Q17, Q18, Q21, Q28, Q39, Q42–Q44 |
| **NO-GO** (future work) | 13 | Q16*, Q20, Q26, Q27, Q29–Q35, Q45 |
| **CLOSED** (already answered) | 3 | Q25, Q35, Q40 |

*Q16 is GO for wiring (1 h, flag off) but NO-GO for training. Q15 similarly split — see entries.

**Headline change from prior plans:** ST baselines and multi-seed MTL start at **3 seeds [42, 123, 7], with a written escalation rule to 5**. The consultation's own protocol docs disagree — Doc 223 §1 demands N=5 for main experiments, Doc 222 §9 tiers full runs at 3 seeds / key ablations 2 / minor 1, and the metrics protocol (`metrics_compilation_2026_07_03`) specifies seeds [42, 123, 7] — while 5-seed ST alone (102.5 GPU-h) would consume 88% of GPU 0's entire remaining budget. Resolution: 3 seeds now; **escalate to 5 seeds iff (a) measured throughput permits (see runtime uncertainty below) or (b) cross-seed variance blurs the MTL-vs-ST deltas** (Doc 223's own escalation logic applied to main runs). Seed semantics per Doc 223: `SEED_DATA=42` frozen project-wide, `SEED_INIT` ∈ {42,123,7}, `SEED_TRAIN = SEED_INIT + 1000`.

**Runtime uncertainty (must resolve Day 1):** the consultation docs price a 100-epoch MTL run anywhere from **~10 GPU-h** (Doc 226 §4.1) to **~50 GPU-h** (FINAL_PAPER_FRAMEWORK §4.5) to **~96 GPU-h** (Doc 222 §1 cost basis, 48 GPU-h per 50 epochs). This ledger uses the conservative 50 h. **Day 1 action: measure min/epoch at epoch 2–5 of the baseline run and rewrite the COMPUTE_SCHEDULE ledger; if a run costs ≤20 h, escalate main MTL to 5 seeds per Doc 223.**

---

## Day-1 pre-launch config review (NEW — blocks all funded runs)

Re-verification against the live code found a launch-critical default the prior plans missed:

1. **`FREEZE_BACKBONE = True` (config.py:199, set 2026-07-07) — linear-probe mode is the current default.** The frozen-ConvNeXt probe ceiling for activity is **0.2169 top-1** (Item 5) against a 0.35 paper target: a main baseline launched as-is is structurally capped below target on activity, and train.py:3808-3816 freezes all backbone params when the flag is True. The paper framework's training spec (§3.2.3: backbone ≈1e-5, heads ≈1e-3) describes *fine-tuning*. **Before launching the Day-1 baseline: decide the mode explicitly — default decision is `FREEZE_BACKBONE=False` (fine-tune at BACKBONE_LR_MULT=0.01) unless the RF1-stage logic is confirmed to unfreeze the backbone itself — and record the decision in the run config.**
2. **Determinism flags** per Doc 223 §1: `torch.backends.cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` in every launch script; bf16 only (fp16 is disallowed — PSR BCE overflows fp16, Doc 211 §1.4).
3. **Fixed split file** referenced (never regenerated); val drives decisions, test evaluated exactly once (Doc 223 §2).
4. Flag audit against the intended run: `USE_KENDALL=True`, `USE_LDAM_DRW=True`, `USE_PSR_TRANSITION=True`, `USE_GEO_HEAD_POSE=True`, `KENDALL_STAGED_TRAINING=False`, `USE_BIFPN/USE_UW_SO/USE_FAMO` per experiment.

---

# CRITICAL — Block Submission

## [Q1] Have we run the ST baselines?
- **Current status:** ❌ NOT IMPLEMENTED (Items 1–4, 66). `scripts/launch_st_baselines.sh` verified complete (dry-run mode, per-head skip, seed loop; internal estimate 102.5 GPU-h @ 5 seeds / 50 epochs on RTX 3060). All four `src/training/train_singletask_*.py` scripts exist.
- **Our decision:** **GO — Day 1, 3 seeds.**
- **Implementation:**
  1. `SEEDS="42 123 7" bash scripts/launch_st_baselines.sh --dry-run` — verify commands (if the script hardcodes 5 seeds, edit its seed list to `42 123 7` first; ~10 min).
  2. Launch on GPU 0 (`DEVICES=0`), order: pose (10.5 h) → activity (15 h) → psr (15 h) → detection (21 h). Pose first: cheapest, unblocks the MediaPipe comparison (Q4) and distillation teachers (Q42) earliest.
  3. Outputs land in `runs/st_baselines/<head>/seed_<n>/metrics.json`.
- **Effort:** 0.5 person-h + **61.5 GPU-h** (GPU 0)
- **Impact:** Enables the paper's core quantitative table (MTL/ST ratio per task). Without it there is no paper.
- **Risk:** 15% — a train_singletask_*.py script has an unexercised bug. Mitigation: 1-epoch smoke per head (`--dry-run` then 1-epoch run, ~2 GPU-h, Day 1 morning) before committing the full launch.
- **Seed note:** 3 seeds now; the two remaining seeds per head are the first claim on GPU 0's ~40 h reserve if Day-1 throughput measurement shows ST runs cheaper than the launcher's estimate (Doc 223 wants N=5; see escalation rule above). Doc 222 §2.2's control — same RNG data-ordering as the MTL run — applies.
- **Dependencies:** Day-1 pre-launch config review (FREEZE_BACKBONE decision applies to ST runs identically — ST and MTL must use the same backbone mode or the comparison is invalid).

## [Q2] Have we run multi-seed main MTL?
- **Current status:** ❌ NOT IMPLEMENTED (Item 67).
- **Our decision:** **GO — 3 seeds, gated behind architecture freeze (Day 21, Aug 3).**
- **Implementation:**
  1. Freeze final config after Phase 2 ablations (see 30_DAY_EXECUTION_PLAN Day 21 gate).
  2. Run `src/training/train.py` full 100-epoch × seeds 42/123/7 sequentially on GPU 1, ~50 GPU-h each.
  3. Bootstrap CIs via existing eval tooling; report mean ± std per the metrics protocol.
- **Effort:** 2 person-h + **150 GPU-h** (GPU 1) at the conservative 50 h/run estimate — could be 30–45 h total if Doc 226's ~10 h/run figure holds (Day-1 measurement decides; escalate to 5 seeds if so).
- **Impact:** Statistical rigor for every headline number; reviewers will demand it. Doc 223 §1 sets N=5 as the protocol minimum for main experiments — the 3-seed floor is a budget-forced deviation that must be (a) escalated away if throughput allows, or (b) defended in the paper via per-sample bootstrap CIs (G9).
- **Risk:** 25% — seed variance large enough to blur claims (Item 64 / Q13). Mitigation: report CIs honestly; the MTL-vs-ST *per-task profile* (which tasks win/lose) is robust to variance even when absolute numbers wobble; variance-triggered escalation adds seeds 4–5 from reserve.
- **Dependencies:** Phase 2 ablation results (Q5/Q41 gates); Q1 not required but desirable for same-protocol comparison.

## [Q3] What is the current detection mAP@0.5?
- **Current status:** 🔴 NEEDS ATTENTION (Item 57, CRIT-1). Repo-root `eval_results.json` is a stale `epoch_0_batch_200` smoke eval (activity 2.2%, pose 30.0°) — useless as evidence. No completed recent 100-epoch run exists.
- **Our decision:** **GO — Day 1.** The measurement comes free from the **main MTL baseline run** (seed 42, current config, 100 epochs) launched Day 1 on GPU 1.
- **Implementation:** Launch current-config run; read `det_mAP50_pc` (present-class protocol, per metrics doc — NOT diluted 24-class mAP) from epoch-checkpoint evals. Epoch-50 checkpoint doubles as the ablation reference (see Q5/Q41).
- **Effort:** 0.5 person-h + **50 GPU-h** (GPU 1, shared with Q9/Q10 measurement)
- **Impact:** Resolves the single largest unknown; target floor mAP50-pc ≥ 0.33.
- **Risk:** 15% mAP ≈ 0 (RISK_REGISTER R1). Trigger: if mAP50-pc < 0.10 at epoch 30, stop, run detection-head diagnostics (`scripts/calibrate_det_threshold.py`, `scripts/eval_detection_dual_protocol.py`) before burning the remaining 35 GPU-h.
- **Dependencies:** None.

## [Q4] Does our pose 8.7° MAE beat MediaPipe?
- **Current status:** 🔵 script exists (`scripts/mediapipe_pose_baseline.py`, verified present), never run.
- **Our decision:** **GO — Day 2.**
- **Implementation:** `python scripts/mediapipe_pose_baseline.py --checkpoint <best ST pose or existing best>` on GPU 0 between ST runs (or CPU). Record forward/up MAE on the identical val protocol.
- **Effort:** 1 person-h + ~2 GPU-h
- **Impact:** Defends the "first head-pose baseline on IndustReal" claim with a non-trivial external baseline. Metrics doc target ≤15°, current claim ~9°.
- **Risk:** 30% MediaPipe wins on angular MAE. Mitigation (paper framing, not compute): MediaPipe is a face-mesh method that fails under industrial occlusion/helmet/viewpoint — report per-condition breakdown and coverage (% frames with a returned estimate), not just mean MAE on MediaPipe's successful frames.
- **Dependencies:** Any decent pose checkpoint (ST pose seed 42 finishes Day 2 on GPU 0).

---

# HIGH PRIORITY

## [Q5] UW-SO loss weighting
- **Current status:** 🔵 Module verified: `src/losses/uw_so.py` (`uw_so_loss(losses, temperature)`, softmax(-loss/T) weighting, stop-grad). Zero references in `src/training/losses.py`, `train.py`, `config.py` — **not wired** (re-confirmed this session).
- **Our decision:** **GO — wire Day 1, ablate in Phase 2.**
- **Implementation:**
  1. `src/config.py`: add `USE_UW_SO = os.environ.get('USE_UW_SO','0')=='1'` and `UW_SO_TEMPERATURE = 1.0` next to `USE_KENDALL` (config.py:47).
  2. `src/training/losses.py`: in the combined-loss path where Kendall weighting is applied, branch: `if C.USE_UW_SO: total = uw_so_loss({'det':L_det,'act':L_act,'psr':L_psr,'pose':L_pose}, C.UW_SO_TEMPERATURE)`. Keep Kendall log_vars registered but unused so checkpoints stay shape-compatible.
  3. Guard: assert not (USE_UW_SO and KENDALL_FIXED_WEIGHTS).
  4. Smoke: 1-epoch run, confirm all four task losses decrease and weights logged.
- **Effort:** 1.5 person-h + 25 GPU-h ablation (50-epoch run vs epoch-50 checkpoint of Q3 baseline)
- **Impact:** V1 Priority-1 recommendation; eliminates Kendall log-var collapse pathology structurally (no cap hack). Expected: PSR/act stability, ±1–3% per-task.
- **Risk:** 20% — softmax weighting over-suppresses the high-loss pose task early. Mitigation: temperature sweep is free at eval time of the same run (log raw losses); T=2.0 fallback.
- **Dependencies:** None for wiring; Q3 epoch-50 checkpoint as ablation reference.

## [Q6] Per-task learning rates
- **Current status:** 🔵 Verified: `train.py:3879` `det_head_lr = head_lr * DET_LR_MULTIPLIER` (config=1.0), `train.py:3882` `activity_head_lr = head_lr * ACTIVITY_LR_MULTIPLIER` (default 3.0). **`psr_params` and pose params get plain `head_lr`** — no PSR/pose multipliers exist in config (train.py:3546-3549 logs them via `getattr(..., None)`).
- **Our decision:** **GO — Day 1 (bundled with Q5 into the "candidate config").**
- **Implementation:**
  1. `src/config.py`: add `PSR_LR_MULTIPLIER = 0.5`, `HEAD_POSE_LR_MULTIPLIER = 0.3` (pose grad norm 3278 = 20,245× PSR's 0.16 per A7 — pose needs *lower* LR, PSR arguably lower too given its tiny gradients are noise-dominated).
  2. `train.py` (~3899): `psr_lr = head_lr * float(getattr(C,'PSR_LR_MULTIPLIER',1.0))`; add a `head_pose_params` split with its multiplier if pose params currently ride in generic `head_params` (verify the param-name filter; pose head params match `head_pose` prefix).
  3. Apply in both Lion and AdamW param-group branches (both exist at train.py:3894+).
- **Effort:** 1 person-h (rides Q5's ablation run — no extra GPU)
- **Impact:** Directly attacks the 20,245× gradient-ratio pathology at the optimizer level; complements UW-SO (loss level).
- **Risk:** 15% — pose underfits at 0.3×. Mitigation: gradient-norm logs (LOG_KENDALL_GRAD_EVERY=500 already in config) show it within 5 epochs.
- **Dependencies:** None.

## [Q7] Task-specific BN (TSBN)
- **Current status:** ❌ Verified absent (no TSBN/task_specific_bn/TaskSpecificBN anywhere in src/).
- **Our decision:** **DEFER — gate: main-baseline det mAP50-pc < 0.33 at Day 8.**
- **If triggered:** Implement per-task affine BN in the FPN/neck (`src/models/model.py`, FPN class near BiFPN at model.py:443): shared running stats, per-task `weight/bias` selected by task-context flag. ~6 person-h + 25 GPU-h ablation.
- **If not triggered:** Future work paragraph.
- **Evidence for deferral:** FINAL §2.5 correction (Item 55): on NYUv2 TSBN *hurt* segmentation (53.93→53.44 mIoU) while helping depth — the "recovers 75% of det gap" claim is not robust across task mixes. Not worth 6 h + 25 GPU-h unless detection is actually the failing head.
- **Effort if triggered:** 6 person-h + 25 GPU-h · **Impact:** up to +5–8 mAP pts if BN-statistics conflict is real · **Risk:** 40% no-effect
- **Dependencies:** Q3 result.

## [Q8] Decoupled activity training (Kang ICLR 2020)
- **Current status:** ❌ Not in standard pipeline; `scripts/decoupled_act_retrain.py` exists as a starting point (verified present).
- **Our decision:** **DEFER — gate: activity clip top-1 < 0.35 (target floor) on main baseline at Day 8.**
- **If triggered:** Stage-2 classifier retrain (cRT): freeze backbone from best MTL checkpoint, re-init activity head, retrain with class-balanced sampling, 10–15 epochs. `decoupled_act_retrain.py` is exactly this — needs checkpoint-path + sampler args wired. ~4 person-h + 5 GPU-h (head-only = cheap).
- **Rationale for DEFER not GO:** LDAM-DRW is already active (`config.py:1147 USE_LDAM_DRW=True`, DRW at epoch 50) and attacks the same long-tail problem. Run one long-tail remedy at a time; measure first.
- **Effort if triggered:** 4 person-h + 5 GPU-h · **Impact:** +3–8% top-1 on tail classes (literature-typical for cRT) · **Risk:** 30%
- **Dependencies:** Q3/Q9 baseline result; best MTL checkpoint.

## [Q9] Activity top-1 current performance
- **Current status:** 🔴 unknown (stale smoke eval shows 2.2% at epoch 0 — meaningless).
- **Our decision:** **GO — measured for free by Q3's baseline run + Q1's ST activity run.** Evaluate with `scripts/eval_activity_75class.py` using **clip-level top-1** (16-frame majority vote per metrics protocol), never per-frame accuracy.
- **Effort:** 0.5 person-h · **Impact:** feeds gates Q8/Q16/Q39 · **Risk:** n/a (measurement) · **Dependencies:** Q3 run reaching ≥ epoch 50.

## [Q10] PSR event-F1 current performance
- **Current status:** 🔴 unknown. Fix verified in place: `config.py:1206 USE_PSR_TRANSITION=True`, `PSR_TRANSITION_SIGMA=3.0` wired at `losses.py:1485`; thresholds HI/LO/sustained in config.py:541-543.
- **Our decision:** **GO — measured for free by Q3 + Q1 runs.** Use `scripts/eval_psr_transition_f1.py` (±3-frame tolerance protocol). Also run the constant-prediction diagnostic (Item 6): all-ones and all-zeros predictors through the same eval — one CPU-hour, establishes the floor the paper must clear.
- **Effort:** 1 person-h · **Impact:** resolves CRIT-3 (60% prior risk of F1 < 0.05) · **Risk:** n/a · **Dependencies:** Q3 run.

## [Q11] Is V1's 312×/140× gradient ratio still relevant?
- **Current status:** ⚪ superseded — A7 re-measured: pose=3278, act=13.80, det=1.86, psr=0.16 → 20,245× (Item 69).
- **Our decision:** **GO — paper edit only.** Cite the new measurement; keep the mechanism narrative (Pathology 3), replace all stale numbers. Grep the paper tex for "312" and "140x".
- **Effort:** 0.5 person-h · **Impact:** correctness of a core paper claim · **Risk:** none · **Dependencies:** none.

## [Q12] ST baselines performance ceiling (HIGH-2 risk)
- **Current status:** 🔴 pending Q1.
- **Our decision:** **GO — same runs as Q1; this is an interpretation contingency, not extra compute.** If ST baselines also perform poorly (35% prior), the paper reframes from "MTL competitive with ST" to "IndustReal is hard in both regimes; we characterize why" — the pathology/diagnosis contribution (Pathologies 1–3) carries the paper. Pre-write both framings in PAPER_OUTLINE §5.
- **Effort:** 0 extra · **Dependencies:** Q1.

## [Q13] Multi-seed result variance
- **Current status:** 🔴 pending Q2.
- **Our decision:** **GO — same runs as Q2.** If cross-seed std > half the MTL-ST gap on any task, report CIs and soften per-task win/loss claims to "parity within noise." No extra compute.
- **Dependencies:** Q2.

## [Q14] Test-val overfitting gap
- **Current status:** ❓ unverified.
- **Our decision:** **GO — one test-split eval at the very end (Day 56–60), after all model selection is frozen.** `scripts/eval_test_split.py` + `scripts/discover_test_subjects.py` exist. Run ONCE on the final 3-seed checkpoints only — never during development (protects against test leakage, which reviewers probe).
- **Effort:** 1 person-h + 3 GPU-h · **Impact:** required for honest reporting · **Risk:** 25% visible gap → report it; per-subject splits make some gap expected · **Dependencies:** Q2 complete.

---

# MEDIUM PRIORITY

## [Q15] ASL for PSR
- **Current status:** 🔵 `src/losses/asymmetric_loss.py` exists; zero references in training path (verified).
- **Our decision:** **Split: GO for wiring (flag off, 1 h) · DEFER training — gate: PSR event-F1 below the STORM anchor (0.506) after Q10 measurement.**
- **Gate calibration warning:** the consultation's target tables disagree — the metrics protocol (Jul 3) targets F1@±3 = 0.50–0.62, while FINAL_PAPER_FRAMEWORK (Jul 14) targets 0.15 (stretch 0.25, fallback 0.05). Two thresholds therefore operate: **< 0.506 → queue ASL for an ablation slot** (this gate); **< 0.30 → RISK_REGISTER R2 fallback chain fires**. Recalibrate both against the Day-5 constant-prediction floor and ST PSR numbers before Day 8.
- **Implementation:** config flag `USE_PSR_ASL`; branch in PSR loss construction in `src/training/losses.py` (~line 1485 region) replacing focal-BCE. If gate fires: 25 GPU-h ablation slot #3.
- **Effort:** 1 person-h + (gated) 25 GPU-h · **Impact:** ASL is designed for exactly PSR's <0.5% positive-rate regime; R3 §3.2 confirmed no published solution at this rate — ASL is the best-evidenced candidate · **Risk:** 35% · **Dependencies:** Q10.

## [Q16] Balanced Softmax for activity
- **Current status:** 🔵 `src/losses/balanced_softmax.py` exists, unwired (verified: zero refs).
- **Our decision:** **Wire behind flag (1 h) · NO-GO for training this cycle.** LDAM-DRW is already the active long-tail remedy; Balanced Softmax is an *alternative*, not additive. Two long-tail A/Bs don't fit the budget; Q8 (cRT) is the better-evidenced second shot if activity underperforms.
- **NO-GO evidence:** Ren et al. (NeurIPS 2020) Balanced Softmax ≈ LDAM-DRW on ImageNet-LT (within ~1 pt); swapping equals-for-equals buys nothing.
- **Effort:** 1 person-h · **Dependencies:** none.

## [Q17] DB-MTL log-transform
- **Current status:** ❌ not implemented.
- **Our decision:** **DEFER — gate: UW-SO ablation (Q5) shows loss-scale imbalance persists (weights saturated at one task for >50% of steps).**
- **If triggered:** one-line change inside the UW-SO branch: `losses = {k: torch.log1p(v) for k,v in losses.items()}` before weighting (DB-MTL arXiv:2308.12029 — corrected ID per Item 56). Rides a Phase-2 ablation slot.
- **Effort:** 0.5 person-h + shares Q5 slot · **Risk:** 25% · **Dependencies:** Q5 result.

## [Q18] Two-stage activity training
- **Current status:** 🔵 partial (`decoupled_act_retrain.py`).
- **Our decision:** **DEFER — merged into Q8.** Same paradigm (decouple representation/classifier); one gate, one implementation. Do not treat as a separate work item.

## [Q19] PSR transition prediction enabled?
- **Current status:** ✅ **RESOLVED this session:** `config.py:1206 USE_PSR_TRANSITION = True` with sigma wiring at `losses.py:1485` and firing thresholds at config.py:541-543. Active by default.
- **Our decision:** **GO — no action beyond documenting in paper §method.** Close the item.
- **Effort:** 0 · **Dependencies:** none.

## [Q20] Per-task augmentation
- **Current status:** 🔵 spatial aug + DetectionAugment exist; full per-task separation absent.
- **Our decision:** **NO-GO — future work.** Engineering-heavy (per-task pipelines through a shared dataloader on frame-aligned labels), no literature evidence of >5% gain for this task mix, and augmentation is already partially task-aware. Fails the decision tree (≥4 h, <5% expected).

## [Q21] TOOD-TAL wiring
- **Current status:** 🔵 `src/losses/tal_assigner.py` exists; **zero references in `src/models/`** (verified).
- **Our decision:** **DEFER — gate: same as Q7 (det mAP50-pc < 0.33) AND Q7's TSBN either not triggered or exhausted.** Detection remedies are rank-ordered: threshold calibration (free) → TSBN → TAL. TAL wiring into `DetectionHead.forward()` is the most invasive (assigner swap touches target building) — last resort.
- **Effort if triggered:** 4 person-h + 25 GPU-h · **Risk:** 35% (assigner swaps are regression-prone mid-project) · **Dependencies:** Q3, Q7 outcome.

## [Q22] Confusion matrix analysis
- **Current status:** 🔵 `scripts/activity_confusion_matrix.py` exists (verified), not run.
- **Our decision:** **GO — Day 9, on the best available activity checkpoint** (main baseline epoch-50+). Produces the paper's error-analysis figure and informs the Q8 gate (are errors tail-class-concentrated?).
- **Effort:** 1 person-h + 1 GPU-h · **Dependencies:** Q3 checkpoint.

## [Q23] Reference code presence check
- **Current status:** ❌ deferred filesystem check. **Cannot be resolved from this repo clone** — `datasets/industreal_github/` does not exist here (verified); it lives on the local workstation (`/media/newadmin/master/POPW/datasets/industreal/`).
- **Our decision:** **GO — 5-minute local action, Day 1:** `ls /media/newadmin/master/POPW/datasets/industreal/` and confirm the authors' eval code (PSR F1 implementation) is present; if absent, `git clone https://github.com/roy-hachnochi/IndustReal` (or the canonical authors' repo). The paper's PSR numbers must be computed with the **authors' scorer** or a verified-equivalent — reviewers check this.
- **Effort:** 0.1 person-h · **Dependencies:** local machine access.

## [Q24] EMA warmup
- **Current status:** ❌ verified: `src/training/ema.py` has no epoch/warmup logic; `train.py:1518` calls `ema.update()` unconditionally.
- **Our decision:** **GO — 0.5 h, Day 2.** Add `EMA_START_EPOCH = 5` to config; guard the `ema.update()` call sites (train.py:1518 and the second site near 2037-2045) with `if epoch >= C.EMA_START_EPOCH`. Prevents epoch-0 garbage weights from polluting the EMA average.
- **Effort:** 0.5 person-h (rides candidate-config ablation) · **Impact:** small (+0–1%) but zero-risk · **Dependencies:** none.

## [Q25] Task head dropout
- **Current status:** ✅ **RESOLVED this session:** pose head has dropout — `src/models/head_pose_geo.py:104` (`dropout: float = 0.1`, applied at :113 and :125). PSR head 0.2 previously confirmed.
- **Our decision:** **CLOSED — no action.**

## [Q26] SWA window expansion
- **Current status:** ❌ no SWA_WINDOW flag (verified absent). `scripts/build_soup.py` exists for checkpoint soups.
- **Our decision:** **NO-GO.** Impact estimate was 8 (lowest tier); soup script already covers the averaging axis at zero training cost. Fails the <5% impact bar. Future work.

## [Q27] Mosaic augmentation activation
- **Current status:** 🔵 verified: mosaic exists **only** in `src/training/pretrain_synthetic.py:159` (pretraining), not the main loop.
- **Our decision:** **NO-GO for main-loop activation — with cause, not neglect.** In a shared-frame MTL setup, a 4-image mosaic destroys the pose target (one head pose per frame), the PSR frame alignment, and activity clip continuity. It is only coherent for the detection stream, which already has `DetectionAugment`. Document the pretraining-only usage in the paper; do not wire into main training.
- **Effort:** 0 · **Dependencies:** none.

## [Q28] OHEM ablation
- **Current status:** 🔵 `DET_OHEM_ENABLED=True` wired (config.py:868, losses.py:328); never ablated.
- **Our decision:** **DEFER — gate: GPU-1 reserve ≥ 50 h on Aug 15 (Day 33) after all higher-priority gates settle.** Single toggle, zero code. If reserve is tight, report "OHEM on (not ablated)" honestly in the paper's training-details table.
- **Effort:** 0.1 person-h + 25 GPU-h · **Dependencies:** budget state at Day 33.

---

# LOW PRIORITY — Future Work (all NO-GO)

## [Q29] Nash-MTL-50 · [Q32] Nash-MTL (full)
- **Decision:** **NO-GO.** PCGrad implemented and active-capable (`src/training/mtl_balancer.py`, mode="pcgrad"). Nash-MTL's Nash-bargaining solve per step is costly and V1 doc 213 rated it complex; with the magnitude pathology (20,245×) addressed by Q5/Q6, direction-surgery upgrades are second-order. Cite as future work with the measured gradient stats as motivation.

## [Q30] CAGrad
- **Decision:** **NO-GO.** Same axis as PCGrad, weaker expected delta than Nash-MTL. One sentence in future work.

## [Q31] Anchor-free detection
- **Decision:** **NO-GO.** V1 doc 212: structural ceiling at 224px input dominates head architecture; `roi_detector.py` (379 lines) stays unwired. Future work: higher-resolution detection stream.

## [Q33] ConsMTL bi-level optimization
- **Decision:** **NO-GO.** Explicitly scoped as "next paper" by V1. Keep it there.

## [Q34] 9D+SVD pose representation
- **Decision:** **NO-GO.** Rejected by V1 doc 215 — marginal over 6D+geodesic, adds SVD backward instability risk.

## [Q35] Geodesic loss replacement
- **Decision:** **CLOSED.** Already at SOTA practice: 6D rotation + huberised geodesic (`src/losses/geodesic_loss.py`, delta=30°). No action.

---

# DATA

## [Q36] PSR per-component positive rate
- **Current status:** ⚪ aggregate <0.5% verified; per-component breakdown missing.
- **Our decision:** **GO — Day 3, CPU-only.** Pandas pass over `PSR_labels_raw.csv` (11 components, NUM_PSR_COMPONENTS per config.py:528): positive rate per component per split. Feeds paper data section + justifies ASL gate (Q15) if some components are ~0%.
- **Effort:** 1 person-h · **Dependencies:** dataset CSV on local machine.

## [Q37] Activity class-0 semantics (conflict RESOLVED by V2)
- **Current status:** ✅ effectively answered — FINAL_VERIFIED_FINDINGS §1.2 verified **class 0 = `take_short_brace` (797 train frames), NOT NA/background** (`ACT_CLASS0_IS_NA = False` per the V1-vs-codebase discrepancy report §4.3); the finding SURVIVED the D1 adversarial challenge. UNANSWERED_QUESTIONS' "CONFLICT" label is stale — V1's background claim was the error.
- **Our decision:** **GO — reduced to a 0.5 h confirmation + documentation task (Day 3).** Confirm `ACT_CLASS0_IS_NA=False` in the loader path actually used at eval time, and state class-0 semantics explicitly in the paper's data section so a reviewer can't misread the confusion matrix.
- **Effort:** 0.5 person-h · **Dependencies:** none.

## [Q38] Body pose annotation source
- **Current status:** ⚪ confirmed pseudo-keypoints from detection boxes (V2 R1 §2.5).
- **Our decision:** **GO — paper documentation only** (one paragraph in data section + limitation note). 0.5 person-h.

---

# ARCHITECTURE

## [Q39] ConvNeXt-Tiny vs MViTv2-S backbone
- **Current status:** ❌ no comparison run. **Correction from the full-archive re-read:** `scripts/train_mtl_mvit.py` / `src/models/mvit_mtl_model.py` exist but are **deprecated MViTv2-S-era dead code** — the staleness report (V2_AGENT_STALENESS_REPORT) documents a deprecation banner at mvit_mtl_model.py:1-11, 13 legacy `scripts/` imports, and that `train.py` does not touch it; the legacy pipeline predates the ConvNeXt migration and every subsequent fix (PSR transition targets, LeakyReLU fix, Kendall clamps, LDAM-DRW). It is NOT a ready-to-launch ablation. V2 FINAL_RANKED_RECOMMENDATIONS explicitly **rejected the backbone swap (Rejected-R2: "committed to convnext_tiny; swap is too risky on timeline")**.
- **Our decision:** **DEFER, now with a higher bar — gate: activity clip top-1 < 0.35 on main baseline (Day 8) AND GPU-1 reserve ≥ 60 h at Day 21 AND ~2–3 person-days available** to either revive the legacy pipeline to parity with current fixes or (cleaner) add an MViTv2-S option to `build_backbone()` in the active `POPWMultiTaskModel`.
- **Effort if triggered:** 16–24 person-h + 50 GPU-h · **Risk:** 45% (legacy revival is regression-prone; 16GB VRAM pressure with MViTv2-S — plumbing smoke first; drop to batch 4 + grad-accum) · **Dependencies:** Q9 result, budget state.
- **Paper note:** if run, report as an ablation, not a second flagship — avoid fragmenting the single-backbone story. If the gate fires but the effort bar isn't met, the cheaper activity remedies (Q8 cRT at 4 h + 5 GPU-h) run first.

## [Q40] 6D rotation correctness
- **Current status:** ✅ column-swap bug at model.py:2177-2178 found and fixed by A11 (`to_legacy_9dof()`); `USE_GEO_HEAD_POSE=True` verified at config.py:1222.
- **Our decision:** **CLOSED.** One regression test exists? — if not, add a 15-min unit test asserting round-trip R→6D→R identity (guards against re-introduction). Include in Day 2 code batch.

## [Q41] Does BiFPN improve over FPN?
- **Current status:** 🔵 verified: `BiFPN` class at model.py:443-540, activation at model.py:1927 behind `USE_BIFPN` — **config.py:160 `USE_BIFPN = False`**. Drop-in interface, zero wiring work left.
- **Our decision:** **GO — Phase 2 ablation slot #2 (Day 10).** `USE_BIFPN=1` env, 50-epoch run, 25 GPU-h, compare det mAP50-pc + pose MAE vs baseline epoch-50 reference.
- **Effort:** 0.5 person-h + 25 GPU-h · **Impact:** +1–3 det mAP typical for BiFPN at this scale · **Risk:** 25% neutral-negative · **Dependencies:** Q3 epoch-50 reference checkpoint.

---

# TRAINING

## [Q42] Does distillation actually help?
- **Current status:** 🔵 hook wired: train.py:1573 (loss hook), :3797-3800 (teacher cache, default `runs/teacher_preds`), `USE_DISTILLATION=False` (config.py:1297). Blocked on ST teachers existing. **Caveat from staleness-report finding A4: the distillation call at train.py:1567+ was assessed as a stub needing ~50–100 lines to fully activate** — budget completion work, not just a flag flip.
- **Our decision:** **DEFER — gate: after Q1+Q2, MTL < ST on ≥2 tasks.** Distillation is the gap-closing tool of last resort, not a default ingredient (V2 T1.2 ranked it Tier-1, but that ranking predates the budget ledger). If triggered: complete the stub, generate teacher caches from best ST checkpoints (~5 GPU-h), one distilled run (~50 GPU-h) — only fits if Phase-3 reserve allows; otherwise future work with the wiring documented.
- **Effort if triggered:** 6–8 person-h + 55 GPU-h · **Risk:** 40% · **Dependencies:** Q1, Q2.

## [Q43] Does FAMO outperform Kendall?
- **Current status:** 🔵 verified: `USE_FAMO` env flag (config.py:48), `famo_step` hook (train.py:2092), `src/losses/famo.py` exists.
- **Our decision:** **DEFER — gate: UW-SO ablation (Q5) inconclusive (|Δ| < 1% on every task vs Kendall baseline).** The loss-weighting axis gets at most two funded runs: Kendall (baseline, free) vs UW-SO (slot #1). FAMO enters only as tiebreaker.
- **Effort if triggered:** 0.2 person-h + 25 GPU-h · **Dependencies:** Q5 result.

## [Q44] Does MetaBalance help gradient starvation?
- **Current status:** 🔵 verified: `USE_METABALANCE` (config.py:55), `mtl_balancer.py` mode="metabalance" implemented (lines 89, 178, 200).
- **Our decision:** **DEFER — gate: after candidate config (UW-SO + per-task LR), gradient-norm logs still show pose/psr ratio > 1000×.** MetaBalance is the strongest remaining magnitude-axis tool; it competes for ablation slot #3 with ASL (Q15) — whichever head is weaker gets the slot.
- **Effort if triggered:** 0.2 person-h + 25 GPU-h · **Dependencies:** Q5/Q6 gradient logs.

## [Q45] Does RotoGrad align gradient directions?
- **Current status:** 🔵 verified: `USE_ROTOGRAD` (config.py:142), integrated model.py:2010, 2306.
- **Our decision:** **NO-GO for training.** Direction-alignment is third in line behind magnitude fixes (Q5/Q6) and PCGrad (already available); RotoGrad's rotation matrices add params + instability risk. Documented as "implemented, not ablated" in reproducibility appendix; future work.

---

# PAPER

## [Q46] Is the contribution still novel (Nardon check)?
- **Current status:** ⚪ **INTERNAL CONTRADICTION found in the consultation archive** — MASTER_VERIFICATION Item 75 / R3 describe Nardon arXiv:2506.15285 as "single-task detection + state tracking, different data," threat **LOW** (A19); but V2_AGENT_STALENESS_REPORT finding **A9 describes the same arXiv ID as a "hybrid CNN-attention head pose estimator with 6 DoF," threat MODERATE, June 2026, no code release**. If A9's description is right, Nardon touches our head-pose novelty claim (Contribution 3), not just adjacent-dataset work. The two accounts cannot both be correct.
- **Our decision:** **GO — escalated to Day 4 MANDATORY: read the actual paper (not the consultation summaries) and record what it is.** Then: (1) write the differentiation paragraph against what Nardon actually does; (2) if it is a head-pose method, soften "first head-pose baseline" to the precise claim (e.g., "first head-pose baseline *on IndustReal* / *within a multi-task industrial system*" — whichever survives); (3) refresh the full novelty search Day 80 (queries in LITERATURE_GAPS G1).
- **Effort:** 3 person-h + 1 person-h refresh · **Risk:** raised to 20% that a claim needs rewording (was 10%) — rewording is cheap if done in July, expensive if discovered by a reviewer · **Dependencies:** none.

## [Q47] Final paper title
- **Current status:** 🟡 draft exists.
- **Our decision:** **GO — freeze at Day 60 (Sep 11).** Recommended: **"Multi-Task Industrial Assembly Perception: A Single-Backbone System for Detection, Activity, Procedure State, and Head Pose on IndustReal."** Revisit only if Q12's contingency framing (both-regimes-fail) takes over, in which case: "...: Pathologies and Remedies for Multi-Task Learning on IndustReal."
- **Effort:** 0.2 person-h · **Dependencies:** Q2 results shape which framing wins.

## [Q48] All 23 R3 citations properly formatted?
- **Current status:** ❓ unverified.
- **Our decision:** **GO — Day 62, 1 h.** Cross-check `popw_aaiml2027.tex` bibliography against R3_LITERATURE_VERIFIED list; verify the DB-MTL arXiv ID reads 2308.12029 (Item 56 correction) and gradient numbers per Q11.
- **Dependencies:** paper draft exists.

## [Q49] FABRIC ATRE benchmark in related work
- **Current status:** ❌ not added.
- **Our decision:** **GO — Day 62, 1 h.** One paragraph in §2 positioning IndustReal vs FABRIC ATRE (task taxonomy differences); strengthens the "why IndustReal" motivation.

---

# TIMELINE

## [Q50] Days remaining and phase plan
- **Current status:** 88 days (Jul 14 → Oct 10).
- **Our decision:** **GO — resolved by this document set.** Revised phases (vs the UNANSWERED_QUESTIONS draft which had 400 GPU-h demand vs ~300 available — resolved by 3-seed protocol):
  - **Phase 0+1 (Days 1–8, Jul 14–21):** all zero-GPU wiring + ST baselines (GPU 0) + main baseline (GPU 1). Gates Q3/Q9/Q10 read at Day 8.
  - **Phase 2 (Days 9–21, Jul 22–Aug 3):** 2–3 ablation slots on GPU 1; gated items resolve; **architecture freeze Day 21.**
  - **Phase 3 (Days 22–49, Aug 4–31):** 3-seed final MTL (GPU 1), remaining evals/figures (GPU 0), paper drafting in parallel.
  - **Phase 4 (Days 50–88, Sep 1–Oct 10):** test-split eval, full paper, internal review, 1-week submission buffer.
- **Budget check:** demand GPU 0 ≈ 76 h (cap 116) ✅ · GPU 1 ≈ 275 h funded + ≤75 h gated (cap 365) ✅ — see COMPUTE_SCHEDULE.md.

---

# Beyond the 50 questions — consultation items promoted into the plan

The full archive re-read (all ~90 files) surfaced three V2 items that are not among UNANSWERED_QUESTIONS' Q1–Q50 but are load-bearing for the paper. They are adopted:

## [X1] Uncapped-Kendall control ablation (V2 T1.6, Doc 222 §3.1)
- **Why it was missed:** not phrased as a question in UNANSWERED_QUESTIONS; but the paper's Contribution 2 ("Kendall-cap configuration resolves collapse") is indefensible without the uncapped control, and FINAL_PAPER_FRAMEWORK requires it twice — ablation Table 5 row 1 AND Figure 2 (log-var trajectories with/without caps). Doc 222 lists it among the *key* ablations.
- **Decision:** **GO — funded ablation slot (not gated).** `scripts/launch_uncapped_kendall.sh` exists; 50-epoch run, seed 42, ~25 GPU-h on GPU 1. Log log-var trajectories every 500 steps (already configured) for Figure 2.
- **Effort:** 0.5 person-h + 25 GPU-h.

## [X2] Code-release / reproducibility track (V2 FINAL_IMPLEMENTATION_PLAN Days 57–65)
- **Decision:** **GO — Days 50–63:** public repo cleanup, README + `requirements_frozen.txt` + reproduce script, checkpoint archival (Zenodo DOI), weights upload. Zero GPU; ~8 person-h. Check AAIML's anonymity policy first (G8) — if double-blind, release is anonymized/post-decision.

## [X3] Cloud-GPU fallback (V2 T3.4, Doc 226 §risk)
- **Decision:** **GO — provision only, $0 until triggered.** Create the RunPod/Lambda account and test a 1-epoch cloud run early (Week 2, ~$5), so RISK_REGISTER R8's fallback is exercised before it's needed. Budget cap $200–500 per Doc 226 (~300 GPU-h on an RTX 4090 at ~$0.34–1.50/h).

Also adopted from Doc 222's compute-saving playbook for all ablation slots: 25-epoch runs for loss-weighting sweeps where ranking stabilizes early, epoch-10 ranking checks for neck ablations, and checkpoint-reuse for on/off switches that diverge late — apply whenever a slot's question tolerates it, and bank the saved hours.

---

# Self-check against decision rules

Every CRITICAL item → Phase 0–1 ✅ · Every GPU-requiring non-critical item → gated behind Phase 1/2 results with explicit trigger ✅ · Every <4 h zero-GPU item → Phase 1 wiring batch ✅ · Every >4 h, <5%-impact item → future work ✅ · Total funded GPU demand ≤ budget with reserve on both GPUs (see COMPUTE_SCHEDULE ledger, incl. X1's +25 h) ✅ · Day-1 pre-launch config review added (FREEZE_BACKBONE, determinism, split file) ✅

**End of AAIML_SUBMISSION_CHECKLIST.md**
