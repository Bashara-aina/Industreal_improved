# 157 — Ultimate Answers: Every Question in Documents 150–156, Verified Line by Line

**Date:** 2026-07-07
**Scope:** This file answers every question, verifies every claim, and audits every file path in the seven strategy documents 150–156:
`150_MASTER_SYNTHESIS.md`, `150_SOTA_STATUS_V5.md`, `151_PER_HEAD_DEEP_ANALYSIS.md`, `152_IMPLEMENTATION_BUG_CATALOG.md`, `153_MULTI_TASK_DEBATE.md`, `154_SOTA_COMPARISON.md`, `155_FINAL_PAPER_NARRATIVE.md`, `156_100_DEEP_QUESTIONS.md`.
**Method:** Every document was read line by line. Every claim that can be checked against this repository (code, evidence JSONs, git history) **was** checked, and the answers below cite the verified evidence. Claims that live only on the workstation (`/media/newadmin/...` absolute paths, `/tmp/*.log` training logs, `best.pth`/`crash_recovery.pth` 738MB checkpoints, GPU process state) are marked **UNVERIFIABLE-REMOTELY** rather than repeated as fact.

**Verification legend used throughout:**

| Tag | Meaning |
|---|---|
| ✅ VERIFIED | Checked in this repository (file/line/JSON/git object exists and matches) |
| ❌ CORRECTED | The document's claim is contradicted by repository evidence; the correction is given |
| ⚠️ PARTIAL | Claim partly verified; the unverified part is stated |
| 🔒 UNVERIFIABLE-REMOTELY | Only checkable on the workstation (logs, checkpoints, running processes) |

---

## §0. Findings of This Audit — Corrections the Documents Themselves Need

Before answering the questions, this audit found twelve facts that change or sharpen answers throughout. Each is repo-verifiable and cited. These are the highest-value content of this file because several of them invalidate expectations that documents 151–156 still carry.

### F-1. ❌ `DETACH_PSR_FPN=False` in the V3 launch script is a **no-op**. The V3 run trains with the PSR FPN **still detached**.

This is the most consequential new finding, and it is the *fourth* exhibit of the documents' own thesis ("code that exists but does not execute is invisible to loss curves"). Evidence chain, all in this repo:

1. `scripts/train_psr_repair_v3.sh` exports `DETACH_PSR_FPN=False` as an **environment variable** (verified in the script, which announces "FIX: DETACH_PSR_FPN=False (was True)").
2. `src/config.py:1072` hardcodes `DETACH_PSR_FPN = True`. A whole-repo grep finds **no `os.environ` read for `DETACH_PSR_FPN` anywhere** — unlike `KENDALL_FIXED_WEIGHTS` (config.py:96, env-read) and `PSR_HEAD_REPAIR` (config.py:105, env-read).
3. The only two mechanisms that can change it are (a) `apply_preset` at config.py:2151 — and **every preset in config.py sets `'detach_psr_fpn': True`**, including `stage_rf4`, the preset the V3 script passes; and (b) the CLI flag consumed at train.py:5713–5715, which can only set it **to True**, never to False (`if getattr(args, 'detach_psr_fpn', False): C.DETACH_PSR_FPN = True`).
4. `scripts/train_psr_repair_wrapper.py` (which the V3 script actually launches) patches only `MIXED_PRECISION`/`AMP_DTYPE`; it does not touch `DETACH_PSR_FPN`.
5. train.py:1384 reads the value from the config module: `_psr_fpn_detached = bool(getattr(C, 'DETACH_PSR_FPN', False))`.

**Consequence:** the in-flight V3 run applies **exactly one** live loss-side intervention — `KENDALL_FIXED_WEIGHTS=1` — plus whatever head-architecture state its process loaded at start (see F-10). PSR gradients still do **not** flow into the FPN/backbone. This revises 152 §1.4 ("The V3 Fix" — the fix is not active), 152 §6 row 10, 153 §1.3, 155's Training Configuration paragraph ("Gradient flow to all heads is enabled via DETACH_PSR_FPN=False" — false for the running V3), 156 Q5 and Q41. Note the nuance: `detach` severs gradient flow *into the FPN/backbone*, not gradients *within the PSR head itself* — so the head-local "activations alive" observation is compatible with the detach still being on, and the LeakyReLU re-init alone explains it.
**Fix (one line):** make config.py:1072 read `DETACH_PSR_FPN = os.environ.get('DETACH_PSR_FPN', 'True') != 'False'`, or set `'detach_psr_fpn': False` in a dedicated V4 preset — and add the value to train.py's config dump so this class of failure is visible in logs.

### F-2. ⚠️ File 156 was corrupted by concurrent overwrites. 90 of 100 questions are recoverable; §2 (Q11–20) was never committed.

The committed `156_100_DEEP_QUESTIONS.md` contains only §3, §5, §6, §7, §8, §10, §11 (60 questions). Git history shows the file was **overwritten** (not appended) by successive section commits made within a 25-second window (16:14:09–16:14:34 JST):

| Commit | Message says | File actually contains |
|---|---|---|
| `03ed3751d` | §1 (Q1–10) | §1 (Q1–10) ✅ recovered |
| `b08e01c76` | §9 (Q81–90) | §9 (Q81–90) ✅ recovered |
| `8c9133a54` | §4 (Q31–40) | §4 (Q31–40) ✅ recovered |
| `88befef07` | §2 (Q11–20) | **§6 (Q51–60)** — mislabeled; §2 content nowhere in history |
| `8c398e79e` → `6a57208ac` | §6, §8, §7, §5, §10 | cumulative §3+§6+§7+§8+§5+§10+§11 (current file) |

**Consequence:** §8 of this file answers all 90 recoverable questions verbatim-recovered, and reconstructs-and-answers the lost §2 ("The Implementation Critique, Q11–20") from its evident source material (file 152). The recovered sections should be re-committed into 156 so the repository is self-consistent.

### F-3. ❌ Document 150's list of never-predicted detection classes is wrong.

150 §2 Q1 says "Five detection classes (1, 2, 3, 14, 23) never fire." Repository evidence says otherwise:

- **Zero-GT classes** (no ground-truth boxes anywhere in the 38k eval): **{1, 2, 3, 14, 15, 23}** — six classes, exactly the keys absent from `d3_full_38k/detection_mAP.json:per_class_AP50` (present keys: 0, 4–13, 16–22 = 18 classes). This also resolves the "6-vs-9 zero-GT discrepancy" (140 §0 row 7): the JSON supports **6**.
- **Never-predicted classes**: **{1, 13, 16, 19, 23}** per 151 §2.2 and 152 §2.1 — and `detection_root_cause/analysis.md` *proves* classes 2 and 3 fire heavily (28,724 and 55,192 predictions respectively **on zero GT** — hallucinations), so they cannot be on a "never fire" list.

**Consequence:** 150 Q1 conflated the zero-GT list with the never-predicted list. Correct taxonomy: classes 1 and 23 have zero GT *and* zero predictions; classes 13, 16, 19 have GT but zero predictions (the real training-convergence failures); classes 2, 3, 14, 15 have zero GT, of which 2 and 3 are hallucinated at volume. Also note 150 Q1's "class mapping bug" attribution is contradicted by 152 §2.2 Fix 4 (commit `a0ffb9aa8`): runtime assertions verified the mapping is **correct**; the never-predicted classes are a **training convergence** issue.

### F-4. ⚠️ Eight cited fix-commit hashes do not exist in this GitHub repository — but the fix **code** does.

`git cat-file -t` finds no objects for `e618d929a`, `6defe1f5f`, `bff38b790`, `216566da0`, `2989801dc`, `96b144e51`, `7001107de`, `a7de2c140` (all cited in 152/155/140). These are workstation-local hashes that never made it here under those IDs. The **code changes themselves are present at HEAD**: PSRHead LeakyReLU repair (model.py:1597–1624), Sequential-index-aware re-init (model.py:1618–1624 checks `head[3]`), head_pose_diag `[6:9]` fix (head_pose_diag.py:85 comment + line 93 uses `[3:6]` only for position), `GuaranteedGTBatchSampler` (industreal_dataset.py:1577), `DET_GAMMA_NEG=2.0` (config.py:832), `FREEZE_BACKBONE`/`BACKBONE_LR_MULT` (config.py:181–182). Six cited commits **do** exist here: `8cef56fc2`, `cd901f655`, `10d5ab596`, `a0ffb9aa8`, `bc6bebdb7`, `28bf668c2`.
**Consequence:** 155's Reproducibility section must be re-keyed to hashes that exist in the public repository before submission, or reviewers who try to check them will find nothing.

### F-5. ⚠️ The decisive D4+D1R 0.6364 was computed on **3 videos** (`d4_d1r/retune/verdict.json: "n_videos": 3`), and its own verdict string is "threshold-partial: decoder shows marginal benefit."

Documents 150/154 present 0.6364 as "the decisive test" without the sample-size caveat. The same documents dismiss the decoder's 0.7893 as "a 2-recording artifact" — a 3-video best-global-threshold number deserves symmetric caution. The D4 (pretrained-YOLOv8m) retune, by contrast, used all 16 videos (`d4_retuned/verdict.json: n_videos: 16`, best-global 0.3466). The direction of the conclusion (detection density binds the decoder) survives, but the 0.6364 magnitude must be labeled "3-video, post-hoc global threshold sweep" wherever it appears.

### F-6. ✅ The four evidence directories flagged "MUST COMMIT" are now committed.

`git ls-files` confirms: `d4_retuned/` (metrics, sweep, thresholds, verdict), `full_eval_ep18_v2/metrics.json`, `up_vector_v3/up_vector_per_recording.json`, `d1_yolov8m_v3/metrics.json`, and the D1R `results.csv` files (`runs/detect/.../yolov8m_industreal/d1r/results.csv`, 25 epochs, and `d1r_proper/results.csv`). Document 150 §0.2's "**Must be committed** — currently not in the repo" for `full_eval_ep18_v2/metrics.json` is **stale — the action item is done**. The 9.14°/7.78° and 0.347 numbers are now auditable from this repo.

### F-7. ⚠️ Internal discrepancy: post-activation value "+384" (150_SOTA_STATUS_V5 line 19: "activations +384 on sequence frames") vs "+4608" (150 §0.3/§1.2, 151 §3.3/§3.5, 152 §1.4, 156 Q5/Q41/Q83).

Both cannot be the same measurement. Neither is auditable remotely (the value lives in `/tmp/train_psr_repair_v3.log` on the workstation). Until the log is committed or quoted with a timestamp, papers should say "post-activation statistics moved from a saturated negative regime to large positive values" without a specific number, or commit the log excerpt.

### F-8. ⚠️ LOO-CV standard deviation: the quoted "+0.0148 ± 0.0158" does not exactly match the committed artifact.

`psr_loo_cv_stratified/loo_stratified.json` records `loo_improvement_mean: 0.0148, loo_improvement_std: 0.0163`, with `loo_global_f1_mean: 0.6710` and `loo_optimal_f1_mean: 0.6858`. Use **±0.0163**. (The earlier `psr_loo_cv` figure "+0.0358 ± 0.0216" in 140 §0 row 11 is superseded by the stratified run.) The conclusion is unchanged: the CI includes zero; per-component threshold tuning is not statistically supported; the honest primary is global-0.10 F1 = 0.6788.

### F-9. 🔒 The following claims are unverifiable from this repository and must be labeled as such wherever they appear:

- `best.pth` and `crash_recovery.pth` (738MB, SHA256 `59cb88ec...`) — **not in the repo**; the SHA can only be confirmed on the workstation.
- `/tmp/train_psr_repair_v3.log` (epochs 24+, post-activation values) and `/tmp/train_singletask_det.log` (epoch 43+, ~3.4 days remaining) — workstation-local.
- Whether the running V3 **process** loaded the repaired PSRHead code (see F-10) — requires the 2-minute workstation check (140 §-1 Blocking Diagnostic #1; still PENDING per `opus_140_141_compliance/audit.md` Gap 5).
- All `/media/newadmin/master/POPW/...` absolute paths in 150–156 resolve on the workstation only; their repo-relative equivalents under this repository are what this audit verified.

### F-10. ✅→⚠️ The "real head repair" **is** wired into the committed tree — unconditionally — but whether the in-flight V3 process is running it is unknown.

150 §0.3/§1.2 states "the head repair (`PSR_HEAD_REPAIR`) was never wired into the execution path." That was true of the **env-gated** path: `PSR_HEAD_REPAIR` (config.py:105) is consumed only by the dead `PSRTransitionPredictor` (psr_transition.py:203–209) and remains a no-op. But the current HEAD carries the repair **hardcoded into the live `PSRHead`**: model.py:1604–1611 has `LeakyReLU(negative_slope=0.01)` in all 11 `output_heads`, and model.py:1618–1624 applies `normal_(std=0.01)` + `zeros_` bias init, under the comment `[REPAIR 2026-07-07 Opus 140 §-1d + diagnostic 96b144e51]`. So the committed tree ≠ the tree 140 §-1 audited (`7001107de`, which is itself not in this repo — F-4). What remains genuinely open: **did the V3 process start before or after this code landed, and did its checkpoint-resume overwrite the re-initialized heads?** That is the workstation check. Until it is done, describe V3 as: "Kendall-fixed weights, PSR-FPN still detached (F-1), head architecture state unconfirmed."

### F-11. ⚠️ Two different "null" baselines are conflated across the documents.

- **copy-prev / persistence null** (predict previous frame's state): F1 = 0.9997 (`null_copy_prev`), POS = 0.9984 (`null_model_pos`, n=5,000 frames, 3 recordings; ours POS 0.9988; all-zeros POS 0.9995).
- **always-positive prevalence null** used in the per-component null-delta table: F1_null = 2p/(1+p) (`psr_null_delta_table.md`), which yields the +0.097/+0.093 deltas on comp4/comp10 and −0.000 on comp9 (and +0.053 on comp8, which the summaries omit).

Sentences like "model is 29.7% worse than persistence" (copy-prev) and "null-delta +0.097" (always-positive) use **different nulls**; a paper table must name each null explicitly or reviewers will find the apparent contradiction.

### F-12. ✅ Single-task ablation presets already exist in config.py — the "not started" baselines are config-ready.

`ablation_det_only` (config.py:~1663, "Ablation A1: detection-only, arch+hparams == stage_rf4"), `ablation_act_only` (A2, ~1694), `ablation_psr_only` (A3, ~1727), `ablation_pose_only` (A4, ~1760), plus `ablation_single_task` (~1621). Also relevant: `pose_multitask_vs_singletask/comparison.md` exists and concludes "Pose is not a confirmed multi-task win… Reserve multi-task claims for tasks where single-task ablation exists," and notes the pose-only preset is **unrun**. Several answers below (150 Q33/Q39, 153 §3.2, 156 Q81/Q82) are sharpened by this: launching a single-task baseline is a one-line preset selection, not new engineering.

---

## §1. Document 150_MASTER_SYNTHESIS — Answered Line by Line

### §1.1 The Evidence Inventory (150 §0) — audit of every listed path

| # | Path (repo-relative) | 150's claim | Audit result |
|---|---|---|---|
| 1 | `src/models/model.py:1539-1640` | PSRHead is the actual trained head; GELU + 0.1 bias; `[AUDIT]` comment; heads at 1609–1611 | ⚠️ PARTIAL — `PSRHead` at 1539 ✅; but HEAD now carries the **LeakyReLU repair** at 1604–1624 (F-10); the GELU/+0.1-bias description is of the *pre-repair* tree; `_debug_log_head0` is at model.py:1648 (not 1635) |
| 2 | `src/models/activity_tcn.py` | ActivityTCN committed, never trained | ✅ VERIFIED (file exists; no training artifacts) |
| 3 | `src/models/activity_tcn_vit.py` | ActivityTCNViT committed, blocked on GPU | ✅ VERIFIED |
| 4 | `src/models/video_backbones.py` | MViTv2-S extractor (Kinetics-400) | ✅ VERIFIED |
| 5 | `src/models/video_backbone_multitask.py` | 53.8M total / 19.3M trainable | ✅ file exists; param counts stated in file docs (not re-derived here) |
| 6 | `src/evaluation/activity_mvit_probe.py` | produced the 0.3810 | ✅ VERIFIED — `activity_mvit_probe/results.json: best_val_top1_69 = 0.381048`, majority 0.266633 (clip-level), improvement +0.114415, best epoch 6 |
| 7 | `src/evaluation/null_model_pos.py` | POS artifact: all-zeros 0.9995, copy-prev 0.9984 | ✅ VERIFIED — `null_model_pos/*.json: 0.99954 / 0.99843`, ours 0.99880 (n=5,000 frames, 3 recordings) |
| 8 | `src/evaluation/convnext_psr_decoder.py` | 0.0053 transition F1, saturated logits | ✅ script exists; number consistent with SOTA_STATUS.md decoder row |
| 9 | `src/evaluation/activity_temporal_probe_cpu.py` | bare-except bug fixed at `7001107de` | ⚠️ script exists and runs; the cited hash is **not in this repo** (F-4); probe result committed: clip mean-pool **0.0723** vs majority 0.2217 (delta −0.1493, best epoch 15) |
| 10 | `src/evaluation/psr_true_signal_analysis.py` | null-delta +0.097/+0.093/−0.000 | ✅ VERIFIED via `psr_null_delta_table.md` (comp4 +0.097 @ p=0.142; comp10 +0.093 @ p=0.183; comp9 −0.000; **comp8 +0.053 omitted from the summaries**) |
| 11 | `checkpoints/best.pth` (738MB, SHA `59cb88ec...`) | recovery checkpoint, must be cold-copied | 🔒 UNVERIFIABLE-REMOTELY — not in repo |
| 12 | `checkpoints/crash_recovery.pth` | secondary recovery | 🔒 UNVERIFIABLE-REMOTELY |
| 13 | `checkpoints/SOTA_STATUS.md` | "contains stale claims… must be edited Day 1" | ❌ CORRECTED — the committed SOTA_STATUS.md is already the **rewritten** 2026-07-07 version: honest labels ("cross-architecture ceiling", "impl bug", "structural artifact", null tables, per-component breakdown). Residues to fix: the D1R row still carries the uncited "~0.95" as SOTA, and pose rows say "uncited". The Day-1 edit is ~90% done |
| 14 | `checkpoints/disclosures_v1.md` | 12 disclosures, superseded by 140 §4's 8 | ✅ file exists |
| 15 | `activity_mvit_probe/results.json` | 0.3810 | ✅ VERIFIED (0.381048) |
| 16 | `full_eval_ep18_v2/metrics.json` | 9.14/7.78 — "**Must be committed** — currently not in the repo" | ❌ STALE — **it is committed** (F-6); contains full per-comp PSR + pose fields |
| 17 | `d3_full_38k/detection_mAP.json` | 0.00009 | ✅ VERIFIED (`full_set_mAP50 = 9.108e-05`); 18 present-class keys; absent {1,2,3,14,15,23} |
| 18 | `psr_optimal_thr_38k/optimal_thresholds.json` | 0.7018 | ✅ VERIFIED (`optimal_macro_f1 = 0.70176`, `global_0.10_macro_f1 = 0.67875`, n_frames 38,036) |
| 19 | `d4_d1r/retune/verdict.json` | 0.6364 decisive | ⚠️ VERIFIED number (0.63636) — but `n_videos: 3` and verdict string "threshold-partial: decoder shows **marginal** benefit" (F-5) |
| 20 | `/tmp/train_psr_repair_v3.log`, `/tmp/train_singletask_det.log` | V3 alive at +4608, epochs 24+/43+ | 🔒 UNVERIFIABLE-REMOTELY; and see F-1 (DETACH no-op) and F-7 (+384 vs +4608) |
| 21 | Strategy files 132–150 table (§0.4) | statuses | ✅ all listed files exist in `analyses/consult_2026_06_10/AAIML/`; 156's own integrity is broken (F-2) |

### §1.2 150 §1 "The Best-Of-The-Best Path" — claim-by-claim verdicts

**Head pose 9.14°/7.78° (§1.1):** ✅ VERIFIED. `bootstrap_ci.json`: forward weighted mean 9.1355°, CI [7.7412, 10.8704]; up 7.78°, CI [6.89, 8.81]; 16 recordings, 38,036 frames, 1,000 bootstrap resamples, seed 42. Per-recording values confirm the outlier (14_assy_0_1 forward 17.0485°). Training-loss indices verified **in this audit directly**: `losses.py:951–952` slices `fwd=[0:3], pos=[3:6], up=[6:9]` for both pred and target. The "first baseline" claim remains gated on the literature search (137 Q42/Q44 — Jiang ECCV'22, HoloAssist NeurIPS'23, Tome CVPR'23), which no committed artifact yet documents: **still open, still blocking the word "first."** Kalman −1.5%/−2.7% ✅ consistent with `pose_kalman_eval` (9.00/7.58).

**D1R 0.995 mAP50 (§1.1):** ✅ VERIFIED from committed `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv`: epoch 25 row reads mAP50 **0.99484**, mAP50-95 **0.86096**, precision 0.996, recall 0.997. Cross-architecture caveat is correct and mandatory. Note the companion `d1r_proper/results.csv` (3 epochs, mAP50 0.858) is a different, shorter run — do not confuse them. The "BEATS WACV 0.95" phrasing should be retired: the citable WACV numbers are 0.838 (annotated) / 0.641 (entire-video) per 154; "~0.95" is the uncited number the audit ordered removed.

**MViTv2-S probe 0.3810 vs ConvNeXt 0.2169 (§1.1):** ✅ VERIFIED (results.json). One correction of interpretation: the MViT probe's majority baseline is **clip-level 0.2666**, so the honest margin is +0.1144 over baseline — not "0.3810 vs 0.2217" (0.2217 is the *per-frame* majority baseline used for the ConvNeXt probe). The comparison across probes also mixes per-frame (ConvNeXt C5) vs clip-level (MViT, 16-frame) protocols; the conclusion (Kinetics features carry action signal, ImageNet features do not) survives because the ConvNeXt *clip-level* TCN probe also fails (0.0723 vs clip majority 0.2288).

**D4+D1R 0.6364 (§1.1):** ⚠️ VERIFIED with the F-5 caveats (3 videos, post-hoc sweep, "marginal benefit" verdict string). The honest sentence: "With dense detections (YOLOv8m fine-tuned, 0.995 mAP50), the MonotonicDecoder reaches 0.6364 event-F1 on a 3-video subset after a global threshold sweep, vs 0.3466 (16 videos) with pretrained-YOLOv8m detections and 0.000 at default thresholds — evidence that detection density, not the decoder, is the binding constraint."

**PSR 0.7018 / 0.6788 (§1.1):** ✅ VERIFIED (JSON above). Bootstrap CI [0.6436, 0.7321] ✅ (SOTA_STATUS/bootstrap_ci). LOO +0.0148 ± **0.0163** (F-8), CI includes zero ✅. The 10k→38k correction story (0.7499→0.7018; 0.7217→0.6788) ✅ consistent across files.

**150 §1.2 "What We Need to Do":** the four items stand, with two revisions: (i) V3's description must change per F-1/F-10 — it is "Kendall-fixed weights, FPN-detach still on, head state unconfirmed," so its expected lift stays +0.01–0.03 and it **cannot** validate the head repair; (ii) "Apply all 9 implementation fixes and relaunch multi-task V4" must add the **one-line DETACH_PSR_FPN wiring fix** (F-1) and a V4 preset with `'detach_psr_fpn': False`, otherwise V4 re-runs the same silent no-op.
### §1.3 The 50 Deep Questions (150 §2) — Ultimate Answers

#### Detection (Q1–Q10)

**Q1. Is D3 mAP=0.00009 caused by multi-task or implementation?**
**Answer: Implementation-dominant, with a structural data factor — but 150's supporting details need two corrections.** `detection_root_cause/analysis.md` decomposes the failure into three compounding, *measured* causes: (i) **empty-frame flooding** — 38,036 frames, 3,102 GT boxes (8.1% of frames have GT), 3,983,551 predictions total (104.7/frame), 96.2% of them on empty frames = ~3.8M automatic false positives that collapse the PR curve before true positives register; (ii) **class confusion** — only 46% of GT boxes have their best-overlapping prediction in the correct class; classes 7+10 absorb 63% of all predictions with 20% of GT; classes 2 and 3 are hallucinated at volume (28,724 / 55,192 predictions on zero GT); (iii) **box regression failure** — mean best IoU 0.234 (< 0.5 threshold); only 44/3,102 GT boxes (1.4%) are detected at IoU>0.5 with the correct class. Corrections to 150's text: the never-fire class list is **{1, 13, 16, 19, 23}**, not {1, 2, 3, 14, 23} (F-3), and "class mapping bug" is refuted — commit `a0ffb9aa8` added runtime assertions proving the mapping correct; never-predicted classes are a **convergence** failure. None of the three measured causes requires multi-task interference; the single-task ConvNeXt run (in flight) is the controlled test that closes the question.

**Q2. Should we run single-task detection to get a clean cost denominator?**
**Answer: Yes — and it is running (🔒 log workstation-only), it is the single most decision-relevant training of the cycle, and its interpretation tree should be fixed now, before the number exists:** ≥0.5 → ConvNeXt-Tiny can detect; multi-task cost is real and computable as (single − multi)/single; the "implementation > multi-task" narrative gets its denominator. 0.3–0.5 → partial capability; report the cost with an architecture caveat. <0.1 → ConvNeXt-Tiny (at this resolution/anchor config) cannot detect regardless of task count, and **every** "multi-task cost" sentence in 150–155 must be deleted, not caveated. Note config support exists (`ablation_det_only`, F-12), so the run is reproducible from the public repo.

**Q3. Are the 4 detection fixes enough to make multi-task work?**
**Answer: They are necessary, not provably sufficient; expected recovery to 0.1–0.3, not to parity.** The four fixes map one-to-one onto measured pathologies: `GuaranteedGTBatchSampler` (industreal_dataset.py:1577) fixes the 8%-positive-batch starvation; `DET_GAMMA_NEG=2.0` (config.py:832) attacks the 3.8M false positives; the anchor audit (`10d5ab596`) ruled anchors *out* (>99% GT coverage at IoU 0.5); the class-index verification (`a0ffb9aa8`) ruled mapping *out*. What no fix yet addresses: the 46%-correct classification confusion among visually similar assembly states and the IoU-0.234 regression quality — both need training-time signal (more positive exposure) that the sampler provides only indirectly. Success criterion for V4: never-predicted set shrinks from {1,13,16,19,23} to ∅ among classes with GT, predictions/frame drops from ~105 toward single digits, and mAP50 clears 0.1.

**Q4. Should D1R be the "main" detection result, not D3?**
**Answer: No. Keep D1R as the labeled cross-architecture ceiling.** The multi-task paper's detection row must be same-architecture (ConvNeXt-Tiny single-task vs multi-task); D1R (0.99484 @ epoch 25, verified) answers "is the task solvable on this data/split" (yes, nearly saturated) and bounds the denominator from above. Any table cell that lets a reviewer read 0.995 as "our system's detection" is a desk-reject risk given the multi-task head's 0.00009.

**Q5. What does WACV's 0.641 mAP mean for our comparison?**
**Answer: It is the like-for-like row and should replace 0.838 as the primary WACV comparison.** 0.641 is WACV's entire-video protocol — the same protocol as our full-38k evals; 0.838 is annotated-frames-only, a strictly easier population. Adopting 0.641 (i) makes D1R's 0.995 a same-protocol beat, (ii) sets the realistic target for single-task ConvNeXt (0.5–0.7 straddles it), and (iii) shrinks the reported multi-task gap by no honesty cost. This is the cheapest narrative improvement in the detection section — a table edit, not an experiment.

**Q6. Are the 5 never-predicted classes a label mapping bug or learning failure?**
**Answer: Learning failure — the mapping hypothesis is already falsified.** Runtime assertions (fix `a0ffb9aa8`) verified DET_CLASS_NAMES has 24 entries with correct index mapping. The corrected list {1,13,16,19,23} (F-3) splits into: {1, 23} zero-GT *and* zero-pred (nothing to learn, nothing hallucinated — benign); {13, 16, 19} have GT but zero predictions — true convergence failures, consistent with the head's collapse onto high-prevalence states (7, 10, 12). The 10-minute class-histogram check 150 proposes is already effectively done by `detection_root_cause/analysis.md`'s prediction-volume table; the remaining question is whether GT-balanced sampling alone recovers {13,16,19} or whether per-class loss reweighting is also needed.

**Q7. Does present-class mAP (0.573) vs diluted mAP (0.358) change the SOTA narrative?**
**Answer: Not yet, and the audit adds a fourth blocker.** The three stated preconditions stand: WACV convention check (DESK, 30 min), zero-GT reconciliation — **now resolved to 6 = {1,2,3,14,15,23} by the d3_full_38k JSON keys (F-3)** — and full-set eval (the 0.573 comes from a 2.6% class-balanced subsample; the *full-set* JSON shows present-class mAP50 = 9.1e-05, i.e., the subsample number does not survive). Fourth blocker: WACV's own ablation table contains a coincidental **0.573** (COCO→Synth-only row, 154 §2.1) — printing our derived 0.573 next to it invites a false-equivalence reading. Verdict: 0.358 is not reportable either (biased sampling, 154 §3.2); on current evidence the only honest multi-task detection number is **0.00009 full-set**, labeled implementation-bounded.

**Q8. Should we report 0.573 (COCO convention) or 0.358 (24-class diluted)?**
**Answer: Neither, until the full-set eval with detection enabled is re-run post-fixes.** Both derive from the biased 250-batch subsample. The full-set artifact (`d3_full_38k`) reports both conventions and they are *identical* (9.1e-05) because the model's predictions are noise — convention choice only matters when there is signal. Decision rule for the future number: match WACV's convention (COCO evaluates classes with GT only — the 30-min DESK check should confirm), state the convention in the caption, and report the other convention in supplementary.

**Q9. Is the GT-balanced sampler the right fix for the 91.9% empty frames?**
**Answer: Right fix for the training-side starvation; explicitly not a fix for inference-side FP density — and the docs' own numbers prove the second problem is the bigger one.** With ~105 predictions/frame across 34,934 empty frames, even a perfectly class-balanced trained model drowns in FPs unless prediction density falls ~2 orders of magnitude. That is the job of `DET_GAMMA_NEG=2.0` (harder negative mining), confidence-threshold calibration, and NMS discipline (soft_nms.py exists in evaluation/). The sampler (`GuaranteedGTBatchSampler`, verified at industreal_dataset.py:1577, replacing non-GT indices with random GT indices) guarantees 100% positive batches — correct for the 8%-positive-batch pathology. Note one risk the docs don't flag: guaranteed-GT batching changes the effective empty-frame prior the model sees at train time vs the 91.9%-empty eval distribution; the negative-mining strength must compensate, or FP density on empty frames may *worsen*. Track "predictions per empty frame" as a first-class metric in V4.

**Q10. Can we beat WACV 0.641 with single-task ConvNeXt-Tiny detection?**
**Answer: Plausible but unproven; treat 0.5–0.7 as a prior, not a promise.** Basis for the prior: ConvNeXt-Tiny is a competent detection backbone in FPN configurations generally, and D1R proves the split is nearly saturable (0.995). Basis for doubt: `d1_yolov8m_v3/metrics.json` shows even a *real-IndustReal-weights* YOLOv8m evaluated under our pipeline scored 0.0004 (sparse 0.1 pred/frame) before fine-tuning — pipeline/protocol friction is real; and the multi-task head's IoU-0.234 regression suggests the detection head design (anchors, strides, resolution) may cap same-backbone performance below YOLOv8m's. The run answers this in ~3.4 days (🔒); pre-register the three-branch interpretation from Q2.

#### PSR (Q11–Q20)

**Q11. Is the LeakyReLU repair + DETACH_PSR_FPN=False enough to recover F1?**
**Answer: The question's premise has shifted twice, and the honest current state is: the LeakyReLU repair is now in the committed tree (F-10), the DETACH fix is a silent no-op (F-1), and neither has produced a validated F1 yet.** Sequence of record: (1) the original "repair" targeted `PSRTransitionPredictor` — dead code, never instantiated (psr_transition.py:188; only `MonotonicDecoder` and `build_transition_targets` are imported from that module); (2) the real head `PSRHead` (model.py:1539) now carries LeakyReLU + small-normal + zero-bias hardcoded (model.py:1604–1624); (3) `DETACH_PSR_FPN=False` never takes effect (F-1). So "enough?" decomposes into: head-local repair (in tree, effect untested), FPN gradient flow (still off — needs the one-line wiring fix), and transformer variance (unknown — the `[AUDIT]` comment and 140 §-1c suspect near-zero-variance transformer output upstream, which no head activation function can fix). Run the 1-hour activation diagnostic (`_debug_log_head0`, model.py:1648; also `psr_head_activation_diagnostic_v2.py` exists in evaluation/) before believing any repair is sufficient.

**Q12. Will V3 produce F1 > 0.78 once the gradient actually flows?**
**Answer: No such expectation is warranted for V3 as launched.** V3's only guaranteed-live intervention is `KENDALL_FIXED_WEIGHTS=1` (env-read at config.py:96, consumed at losses.py:1666): expected +0.01–0.03, i.e., F1 0.71–0.74 from the 0.7018 base. The 0.78+ projection belonged to the head-repair-active scenario; whether V3's process even has the repaired heads is the pending workstation check (F-10). Decision rule when V3 plateaus: F1 0.71–0.74 → Kendall fix worked as predicted, proceed to the properly wired repair run; ~0.70 flat → publishable negative ("task weighting was not the binding constraint"), which *strengthens* the head-starvation story; >0.78 → either the process did load the repaired heads (check first) or the Kendall effect was underestimated — attribute only after the workstation check.

**Q13. Is Ours F1 (0.7018) > null_copy_prev (0.9997) really "no learning"?**
**Answer: No — but state both nulls correctly (F-11).** Against the **persistence null** (copy-prev, 0.9997), the model is 29.7% relatively worse — a true and reportable fact that reflects PSR's extreme temporal autocorrelation (state changes are rare; persistence is nearly unbeatable frame-wise). Against the **prevalence null** (always-positive, F1=2p/(1+p)), the committed table shows genuine learned signal exactly where it matters: comp4 +0.097 (p=0.142), comp10 +0.093 (p=0.183), comp8 +0.053, vs ceiling-limited ≈0 deltas on the p>0.9 components and a true zero on comp9. Additionally `null_model_pos` proves POS is structurally uninformative (all three of ours/all-zeros/copy-prev ≈ 0.998–0.9995). Verdict: "learned real signal on low-prevalence components; frame-level F1 is persistence-dominated; POS retired to a footnote."

**Q14. Should we report F1 relative to copy_prev, not absolute?**
**Answer: Report three columns per component: absolute F1, the named null, and the delta — and lead the PSR table with the delta columns** (140 Q2.08 concurs). Absolute F1 (0.7018 per-comp / 0.6788 global) keeps comparability for future work; the always-positive null-delta is the honest learning measure; copy-prev belongs in the text as the paradigm caveat, not as a table denominator (dividing by 0.9997 manufactures a misleading "29.7% worse" headline for what is a metric pathology). Include comp8 (+0.053) — the current summaries cherry-pick comp4/comp10/comp9.

**Q15. Is the MonotonicDecoder F1 (0.0053 full-38k) the right comparison?**
**Answer: It is a diagnostic, not a comparison.** 0.0053 measures the decoder fed by saturated ConvNeXt-head logits — it characterizes the checkpoint, not the decoder. The decoder's capability is bounded elsewhere: 0.3466 (16 videos) with pretrained-YOLOv8m detections, 0.6364 (3 videos, F-5) with D1R detections, 0.7893 (2 recordings — artifact). Placement: one row in the PSR section labeled "decoder on saturated logits (diagnostic)," with the D4/D4+D1R ladder in the transition-paradigm table. Never present 0.0053 beside WACV B3 0.883 as if commensurate.

**Q16. Should we use the decoder (0.7893) or the head (0.7018) for the paper?**
**Answer: The head, and the 0.7893 must not appear as a headline anywhere — SOTA_STATUS already demotes it ("earlier 0.7893 was a 2-recording artifact").** The head's 0.7018 (per-comp optimal, with the 0.6788 global-threshold primary per F-8/LOO) is the model's own output on raw video, full 38k, val-selected. The decoder numbers are post-processing under threshold sweeps on small subsets. One table, clear labels, per-frame paradigm and transition paradigm in separate sub-tables.

**Q17. Does the procedure-order constraint (MonotonicDecoder) hurt assembly detection?**
**Answer: Yes — and unlike 150's framing ("has not been quantified"), a bound already exists in the record:** 156 Q45 cites the decoder-oracle analysis — oracle with procedure order ON: 0.5947; order OFF (relaxed): 0.8807 — a ~32-point gap attributable to the hardcoded monotonic chain, with comp4 never placed in 10/16 recordings as the concrete mechanism. What remains unquantified is the *GT order-violation count* across recordings (135 Q43, 30-min CSV pass), which would convert the oracle gap into a dataset-level statement. Until then the decoder limitation sentence should cite the oracle numbers.

**Q18. Is GELU→LeakyReLU the right fix, or a different activation?**
**Answer: Conditionally right; the diagnostic decides, and the mechanism ordering matters.** If per-head pre-activations sit at mean ≈ −130 (as measured), *any* non-saturating negative-slope activation (LeakyReLU) restores gradient where GELU passes ~zero — the fix is right at the head level, and the re-init (small-normal, zero bias) is arguably the more important half since it relocates pre-activations near 0. But if the upstream cause is collapsed transformer output variance (`encoded.std() ≈ 0`, the `[AUDIT]`-suspected condition), heads see near-constant inputs and no activation choice conjures discriminative signal — the repair must move upstream (LayerNorm placement, per-frame-MLP init, or transformer LR). Order of operations: 1-hour diagnostic → if variance healthy, head repair suffices; if collapsed, add a variance-restoring intervention and re-run the diagnostic before training.

**Q19. Can the gradient-DEAD bug (RMS=0.00) be fixed by warm-start, not just detach?**
**Answer: Warm-start and detach address different links in the chain; the answer is "warm-start with repaired init, and separately wire the detach fix."** The zero-RMS gradients had two independent sufficient causes: (a) head-local GELU saturation (pre-activations −130 ⇒ ~zero slope) — fixed by re-init + LeakyReLU, and warm-starting the *old* head weights would reproduce the crash, so the repair's re-init must win over checkpoint-resume for those parameters (worth an explicit resume-time exclusion for `output_heads`, which no document confirms exists — audit gap); (b) `DETACH_PSR_FPN=True` severing PSR→FPN flow — which is *still on* (F-1) and is about representation shaping (whether PSR can adapt backbone features), not about head-local learning. A run can learn head-locally with detach on; it cannot reshape features. Both fixes, one at a time, per single-factor discipline.

**Q20. What's the expected F1 after V3 completes 3–5 epochs?**
**Answer: 0.71–0.74 at global-0.10, with the abort criterion val F1 < 0.65 on two consecutive evals (140 Day 4–7).** Rationale: Kendall-only lift +0.01–0.03 on base 0.7018 (or 0.6788 global — quote the same threshold convention before/after, another place the docs are loose). If the workstation check reveals the process *did* load repaired heads, re-baseline expectations to 0.74–0.80 and re-label the run "Kendall+head-repair bundle" — losing single-factor attribution (the thing 140 §-1b celebrated), which is why the check must happen before the result is interpreted, not after.

#### Activity (Q21–Q30)

**Q21. Is MViTv2-S (Kinetics) the right backbone for assembly activity?**
**Answer: Yes — the probe evidence is decisive and now fully committed.** MViT frozen probe 0.3810 (clip-level, +0.114 over the 0.2666 clip majority baseline; results.json) vs ConvNeXt frozen probe 0.2169 ≈ 0.2217 per-frame majority (CI ±0.0046, statistically null) vs ConvNeXt temporal probe 0.0723 (below baseline — mean-pooling frozen ImageNet features actively hurts). Three independent probes triangulate: the pretraining domain (Kinetics action semantics), not the head, is the binding constraint. Caveat for the paper: 0.3810 is 61% of the 0.6223 protocol-matched SOTA — "right backbone" ≠ "sufficient backbone"; fine-tuning is the required next step.

**Q22. Should we fine-tune MViTv2-S or use TCN+ViT on top?**
**Answer: Fine-tune MViTv2-S; TCN+ViT is justified only ON MViT features, and is dead on ConvNeXt features.** The ACT-1 gate (probe > 0.30) passes for MViT (0.3810) and fails for ConvNeXt on both probes (0.2169 linear, 0.0723 temporal — the TCN-on-ConvNeXt question is already answered negatively, not "pending" as some 156 entries say). Sequencing: fine-tune MViT single-task first (expected 0.45–0.55), and only then consider TCN+ViT temporal aggregation on the fine-tuned features (expected +0.05–0.10). The "one model" narrative cost is real: MViT cannot share the ConvNeXt trunk — the paper should present activity as a per-task backbone-correction finding, not hide the second backbone.

**Q23. Will MViTv2-S fine-tuning close the 0.3810 → 0.622 gap?**
**Answer: About half, and there is committed per-class evidence for both the upside and a specific risk.** Upside: the probe recovers previously-zero classes with big margins (check_instruction 0→0.877 with 529 samples, tighten_nut 0→0.715 — `mvit_per_class/comparison.md`), showing the features linearly separate the high-support classes. Risk the summaries under-report: the frozen probe has only **12 non-zero classes** vs ConvNeXt's 22, **43 classes still at zero**, and a *lower macro mean per-class accuracy* (0.0545 vs 0.0580) — its top-1 win is majority-weighted. Fine-tuning must lift the tail, not just the head; report macro-F1 beside top-1 or the "0.45–0.55 near-SOTA" claim will be vulnerable. WACV's remaining edge (0.6223) plausibly comes from full fine-tuning + augmentation + RGB+VL/stereo ensembling (65.25% published used ensembles, per 154 §1.1).

**Q24. Why does ConvNeXt frozen probe = 0.2169 ≈ baseline (no signal)?**
**Answer: Because per-frame ImageNet-1k features encode object/scene appearance, which in a fixed assembly workspace is nearly constant across activities.** The statistical statement is exact: 0.2169 vs majority 0.2217, 95% CI ±0.0046 — indistinguishable. The mechanistic statement: activity classes in IndustReal differ mainly by *hand-object motion patterns over time* (verb structure), which a single frame's C5 GAP embedding does not carry — and which even *temporal pooling* of such frames does not recover (0.0723 < baseline), because pooling static-appearance features averages away the only weak per-frame cues. This is a structural feature-type mismatch, not an optimization failure — the correct null-result framing for the paper.

**Q25. Can TCN+ViT ever work on ConvNeXt features?**
**Answer: No — and this is already answered by committed evidence, not pending.** `activity_temporal_probe_cpu/results.json`: clip mean-pool top-1 **0.0723** vs clip-level majority 0.2288, delta **−0.149** (best epoch 15 of the probe). Temporal integration of zero-signal features produced a result *below* baseline. The TCN+ViT-on-ConvNeXt line of work is closed; the architectures (activity_tcn.py, activity_tcn_vit.py) remain valuable only as heads over MViT features. Documents that still say "TCN on MViTv2-S: pending" (156 Q26/Q56) are correct that the MViT variant is untested.

**Q26. Is 41/69 zero-accuracy classes evidence of class collapse or backbone mismatch?**
**Answer: Both, causally ordered: backbone mismatch ⇒ no usable per-frame signal ⇒ optimizer collapses onto the prevalence prior ⇒ 41 zero classes.** The discriminating evidence is the probe pair: if collapse were a head/loss artifact (e.g., missing class weighting), the *linear probe* on the same frozen features would still separate some minority classes — it doesn't (0.2169 ≈ majority). Under MViT features, 11 of the 41 zero classes immediately become non-zero under an identical linear head. Residual truth in "collapse": 43 classes are still zero even under MViT probe — prevalence and probe capacity also bind; fine-tuning + class-balanced loss is the follow-up.

**Q27. Should we report per-class activity breakdown in the paper?**
**Answer: Yes — supplementary table with: per-class F1 (top-10 + tail summary), macro-F1 beside top-1, majority-class vs minority-class accuracy split, top-5, confusion matrix with transition-distance histogram, and the verb-antonym error rate (1.3% of errors, already measured per SOTA_STATUS) — plus the MViT-vs-ConvNeXt per-class delta table (`mvit_per_class/comparison.md`), which is the single most persuasive activity artifact:** it converts "backbone swap helps" from an aggregate claim into named, interpretable class recoveries (check_instruction +0.877, tighten_nut +0.715, plug_objects +0.356). The verb-only remap (30 min) remains worth running to test the "actions vs objects" hypothesis.

**Q28. Does MViTv2-S fine-tuning need to be single-task or multi-task?**
**Answer: Single-task first, unambiguously.** It establishes the activity upper bound with one variable changed (backbone), keeps the 2-GPU-week budget defensible, and produces the number the paper needs (activity near-SOTA is claimed *for the backbone correction*, not for multi-task). The multi-task MViT integration (`video_backbone_multitask.py`, 53.8M/19.3M params) is the post-submission experiment; 156 Q59's projection (multi-task 5–10% below single-task) is a reasonable prior but pure speculation until the single-task anchor exists.

**Q29. What's the right balance between backbone pretraining and head architecture?**
**Answer: For this dataset the measured answer is: pretraining domain first (≈ +0.16 top-1 from ImageNet→Kinetics under an identical linear head), fine-tuning second (projected +0.07–0.17), temporal head third (projected +0.05–0.10, unproven), per-frame-head-design last (measured ≈ 0 on wrong features).** The evidence: identical linear heads on two backbones differ by 0.164; the *best* head on the wrong backbone (temporal pooling) *lost* 0.15 vs baseline. This ordering is itself a publishable finding for practitioners: budget goes to pretraining alignment before head engineering.

**Q30. How do we honestly report multi-task activity = 0.0236 vs single-task video = 0.3810?**
**Answer: As the three-act probe/null-result subsection (140 Decision 4), with each act's protocol labeled:** (1) attempt: per-frame MLP on shared ConvNeXt in multi-task → 0.0236 (69-class per-frame; floor); (2) diagnosis: frozen-feature linear probe → 0.2169 ≈ 0.2217 majority (null — the backbone, not the head or the multi-task setting, lacks signal; corroborated by the temporal probe's 0.0723); (3) confirmation: identical probe on Kinetics MViTv2-S → 0.3810 (clip-level; +0.114 over its 0.2666 clip baseline). State per-frame vs clip-level explicitly in every row (the baselines differ: 0.2217 vs 0.2666); do not present 0.0236 and 0.3810 in the same column without protocol flags. The multi-task 0.0236 is never SOTA-compared; it is the pathology exhibit.

#### Head Pose (Q31–Q40)

**Q31. Is 9.14°/7.78° a real "first baseline" claim?**
**Answer: Structurally yes; procedurally still gated.** Verified: numbers, CIs, 38,036 frames, 16 recordings (bootstrap_ci.json); training-loss index correctness (losses.py:951–952, checked directly in this audit); scope pinned to head *orientation* (position is explicitly non-reportable — units unverified vs HoloLens export, 154 §2.4). Still ungated: the documented literature search (Jiang ECCV'22, HoloAssist NeurIPS'23, Tome CVPR'23 under comparable ego-protocols) has no committed artifact. Until a search report is committed, the paper may write "we are not aware of a published ego-pose baseline on IndustReal" but not the bare "first." The uncited ~15° must not appear (154 §2.4: "Do not cite — source unverifiable"); note 155 currently violates this (see §7).

**Q32. Per-recording median (8.94°) or weighted mean (9.14°) as headline?**
**Answer: The audit record contains a genuine disagreement here, and it should be resolved as: weighted mean as primary, median beside it, exclusion variant in supplementary.** 150 argues weighted-mean-primary (reflects per-frame experience); `pose_multitask_vs_singletask/comparison.md` argues median-primary ("robust to the 14_assy_0_1 outlier; report median and IQR, not mean"). Resolution: primary = 9.14/7.78 weighted mean with bootstrap CI (it is the harder, more honest statistic and the CI already prices in the outlier); co-report per-recording median of means 8.94/7.58; 8.46/7.39 (excl. outlier) only with the outlier analysis. **Never use 5.82°** as the up-vector headline — it is the 9-recording median-of-medians covering the easier recordings (140 §0 row 3 explicitly forbids it); note 151 §1.1 and 154 §2.4 still print 8.94/5.82 as a pair, mixing statistics (see §9 ledger).

**Q33. Should we run single-task pose ablation to verify multi-task helps?**
**Answer: Yes as idle-GPU filler; mandatory only if any multi-task-benefit sentence survives editing.** The committed comparison file already made the policy decision: without the ablation, *all* multi-task-attribution language is removed, and pose is presented as a first baseline only. It also bounds the stakes: best recordings cluster at ~6° (24_assy_2_4 6.07, 26_assy_1_5 6.08, 05_assy_0_1 6.26) suggesting a 5–6° floor from GT noise; if single-task landed there, multi-task 9.14° would read as *negative* transfer — a risk the current narrative should acknowledge rather than discover in review. The preset (`ablation_pose_only`, config.py:~1757) exists; cost ~1 GPU-day (F-12).

**Q34. Is the 14_assy_0_1 outlier model failure or data quality?**
**Answer: Model failure — the committed analysis is consistent across files:** GT clean, motion below average, likely visual domain shift (150_SOTA_STATUS_V5; pose_outlier_analysis.md exists in evaluation/). Magnitudes verified: 17.05° fwd / 12.32° up vs next-worst 11.49° fwd. Policy verdict: include in all headline aggregates; report the exclusion variant beside; headline-exclusion would be justified only by a documented GT artifact, which the pose.csv tracking-confidence check (30 min, Day 2) could still surface — run it, but expect it to confirm inclusion.

**Q35. Can pose be improved with single-task training?**
**Answer: Expected modestly, bounded by two committed observations: the ~5–6° per-recording floor (GT noise + sparse-rig ambiguity) and forward-vs-up error correlation r=0.67 (shared cause, likely feature-level).** So the plausible single-task range is 7–8° forward — an improvement over 9.14 but not transformative (the comparison file reaches the same "marginal at best" reading of multi-task benefit). The pose linear probe (137 Q16, ~1 GPU-hr) is the cheap headroom bound and should precede any 1-day training commitment.

**Q36. Report pose before or after Kalman smoothing?**
**Answer: Both, single-frame primary — settled and verified.** Kalman (RTS, Q=0.005, R=0.2): 9.00/7.58, i.e., −1.5%/−2.7% — small because predictions are already smooth; largest gains concentrate on the worst recordings (05_assy_2_2 +0.38/+0.80). One main-text sentence, per-recording table in supplementary. The smoothing result doubles as evidence of temporal consistency — worth one clause, since it preempts "is it flickering?" reviewer questions.

**Q37. Is the 26.20°→7.78° index bug evidence of measurement failure?**
**Answer: Yes — eval-only measurement failure, now triply verified, and it is the cleanest of the three monitoring-blind-spot exhibits.** Verified in this audit: losses.py:951–952 correct at training time; head_pose_diag.py fixed (`[3:6]` now used only for position, line 93) and deprecated in favor of eval_pose_kalman.py. The correct paper-side lesson (155 already states it): 3.5 months of a 3.4× inflated metric survived because no per-task sanity bounds existed on eval outputs. Add the concrete guardrail to the repo: assert up-vector MAE < 45° in full_eval, or unit-norm-check the slice being read.

**Q38. Does the pose loss dominate the multi-task (per Opus A-6)?**
**Answer: No.** Pose works because its head is the simplest readout (two GAP branches → linear, model.py:1530–1533 vicinity) over features ImageNet already provides — not because it starves others. The gradient-suppression mechanism that existed (ACTIVITY_GRAD_BLEND_RATIO=0.05) throttled *activity*, and is now 1.00 in config (config.py:1035 — the fix is live in tree). FiLM verified as static ≈2× scaling (gamma mean 1.98, per-sample std 0.002, dev-from-1 L2 27.7; film_gamma_beta.json) — i.e., no input-dependent cross-task modulation is occurring, for any head. The honest sentence: "no head dominates through FiLM; detection/activity failures are traceable to their own pathologies."

**Q39. Can we beat 9.14° with single-task pose (expected 5–7°)?**
**Answer: The 5–7° expectation is optimistic against the committed floor analysis (~5–6° appears to be the GT-noise floor; realistic single-task ≈ 7–8°).** The narrative consequence in 150 is correct and important: if single-task materially beats 9.14°, pose flips from "multi-task validation" to "multi-task-neutral or hurt" — which is precisely why the comparison file already stripped multi-task-benefit language preemptively. That preemptive framing means the single-task result, whatever it is, cannot damage the paper — a well-constructed position; keep it.

**Q40. Is "first ego-pose baseline" defensible without a published comparison?**
**Answer: Yes — first-measurement claims need completeness of search, not a comparison target.** Requirements, restated as a checklist: (i) commit the literature-search report (present the three named works and why each is protocol-incomparable — face-based, different sensors, different data); (ii) scope the claim to "ego head-orientation on IndustReal under this protocol"; (iii) remove ~15° everywhere (already absent from the rewritten SOTA_STATUS.md ✅; still present in 151 §1.4/155 ❌ — see §9); (iv) report position as non-reportable with the unit-verification reason. With those four, the claim survives review.

#### Cross-Cutting (Q41–Q50)

**Q41. What's the best single experiment to prove multi-task helps?**
**Answer: There is no *single* experiment — "multi-task helps" is per-head — but the highest-information pair is (single-task ConvNeXt detection) × (full-set multi-task detection re-eval post-fixes), because detection is the head with the largest claimed cost and the only one with a same-backbone denominator in flight.** The V3 PSR run adds a second, weaker axis (weight-configuration sensitivity), further weakened by F-1/F-10 (attribution unconfirmed). Honest framing: the achievable claim this cycle is "we can *measure* the multi-task cost per head with controlled denominators," not "multi-task helps." Proving *help* would need a head where multi-task > single-task with all fixes — currently zero heads have that evidence, and pose (the only working head) has an explicit "not a confirmed multi-task win" memo (F-12).

**Q42. Should we cut activity from the paper?**
**Answer: No — the three-act arc (fail → diagnose → confirm) is the methodological contribution, and it is now fully evidence-backed (0.0236 → 0.2169≈null + 0.0723<null → 0.3810).** Cutting it would also orphan the paper's best per-class artifact (mvit_per_class deltas). Requirement: label it a probe/null-result subsection, never a SOTA row; keep the majority baselines printed beside every number.

**Q43. How do we honestly report 3 of 4 heads failing while single-task detection beats SOTA?**
**Answer: As a measurement-and-pathology paper with an explicit two-column structure per head: "what the task supports on this data" (ceilings: D1R 0.995; MViT probe 0.3810; copy-prev 0.9997 as paradigm ceiling context; pose ~6° floor) vs "what our multi-task system achieved and why" (0.00009 — three measured causes; 0.0236 — backbone mismatch; 0.7018 with null-delta-verified signal; 9.14/7.78 working).** The contribution claim: verified failure modes, the corrected mechanism (GELU starvation in `PSRHead.output_heads`, not the dead class), concrete fixes with commits in the public repo, and the monitoring thesis with (now) four exhibits: dead PSRTransitionPredictor, NaN checkpoint selection, up-vector eval index, and the DETACH_PSR_FPN env no-op (F-1) — the fourth being *discovered by the auditing method the paper advocates*, which is the strongest possible demonstration of the thesis.

**Q44. Is "Implementation > Multi-Task" the right paper title?**
**Answer: No for the submission; keep it as the internal thesis slug.** It pre-announces the conclusion of an ablation not yet run (single-task denominators pending) and undersells the first baselines. 155's actual title — "What Four Tasks Cost One Backbone: A Pathology Analysis of Multi-Task Training on IndustReal" — is better and already adopted; it stakes a measurement claim the freeze-date evidence can support. Second-choice framing if the single-task denominator lands well: "Measuring What Four Tasks Cost One Backbone."

**Q45. Should we report DETACH_PSR_FPN=False as a paper contribution?**
**Answer: Not as a contribution — as a pathology exhibit, upgraded by this audit:** the flag's *intended* flip never executed (F-1). One paragraph in the pathology section: default True (config.py:1072, gradient-isolation rationale in the comment), flip attempted via env var, env never read, preset and CLI paths can only reinforce True, and the training log prints no value for it (the config dump omits it — the observability gap that let it slip). That paragraph *is* the paper's thesis in miniature.

**Q46. What's the right balance between pathology paper and SOTA-comparison paper?**
**Answer: Pathology-dominant, ~70/30, with the venue thresholds from 140 §5 unchanged:** PSR ≥ 0.78 with a properly attributed repair + clean detection denominator → AAIML main track; PSR 0.72–0.78 → short/workshop; detection denominator broken → arXiv-first. Post-audit adjustment: the realistic Jul 20 bundle (V3 ≈ 0.71–0.74, single-task det pending, MViT probe only) sits at the **short/workshop-to-main-track boundary**, with the deciding factor being the single-task detection number, not PSR.

**Q47. Should the paper emphasize single-task wins or multi-task failures?**
**Answer: Neither in isolation — emphasize the *gap* and its decomposition.** The rhetorical structure that survives review: each head gets (ceiling, achieved, gap, cause, fix, post-fix status). D1R 0.995 is only meaningful as the numerator's ceiling; 0.00009 only as the pre-fix floor; the paper's value is the causal account connecting them. The one-sentence summary for the abstract: "the gap between what this data supports and what a shared-backbone system achieved is, for three of four heads, attributable to specific, fixable implementation defects — which we identify, fix, and re-measure."

**Q48. Is the AAIML submission worth it given current numbers?**
**Answer: Yes — with the asset list post-audit:** first ego-pose baseline (gated on the search report), first per-frame PSR baseline with prevalence-null-verified signal, the D4/D4+D1R decoder-transfer ladder (with the n=3 caveat), the MViT probe breakthrough with per-class recovery table, D1R same-protocol ceiling 0.995, three (now four, F-1) documented pathology exhibits, and — pending — the single-task ConvNeXt denominator. That is a coherent measurement paper. Deadline risk flagged in 150 §3.2 item 12 stands: **the AAIML deadline is documented nowhere in the repository** — confirming it is a 5-minute action that gates the entire schedule and should have been done before planning a Jul 20 freeze.

**Q49. How do we handle the in-flight training results?**
**Answer: Per 150's schedule, with three amendments.** (i) V3's product is re-labeled (Kendall-only, F-1/F-10) and its go/no-go remains the 0.65 abort floor; (ii) the single-task detection checkpoint must be evaluated with **the same full-38k protocol** as D3 (same conventions, same eval code — `results_frozen.json` discipline per 138 Attack 10) or the cost ratio will be attacked as protocol-mismatched; (iii) cold-copy verification of `best.pth` (SHA `59cb88ec...`) is workstation-only — do it before any V4 launch overwrites run dirs, and record the copy's hash in a committed file so the freeze is auditable from the repo.

**Q50. What's the absolute best case if all 9 fixes work + V3 trains + MViTv2-S fine-tunes?**
**Answer: 150's own two-tier answer is correct and survives audit, with expectations adjusted:** Best case (needs: fixes-applied V4 relaunch *including the F-1 wiring fix*, properly attributed PSRHead repair run, 2 GPU-weeks of MViT fine-tuning, 4 single-task baselines): detection-MT 0.3–0.5, PSR 0.78–0.84, activity 0.45–0.55, pose 9.14/7.78, detection-ST 0.5–0.7 — feasibility before Jul 20: **low** (MViT fine-tune alone consumes the calendar). Realistic Jul 20 case: detection-MT 0.1–0.3 (if V4 launches immediately after the in-flight runs), PSR 0.71–0.74 (Kendall-only), activity 0.3810 (frozen probe), pose 9.14/7.78, detection-ST 0.5–0.7 (in flight). That bundle, honestly framed, is a solid AAIML pathology/measurement submission — and every number in it except the two in-flight results is already committed and auditable in this repository.

### §1.4 150 §3 (Implementation Path) — assessment

Day 1–3 items: sound, with V3 re-labeled (F-1/F-10). Day-1 hygiene list (§3.2): items 2 (commit evidence dirs) and 3 (SOTA_STATUS edit) are **already substantially done** (F-6, §1.1 row 13); item 1 (workstation no-op check) remains the top open blocker; item 5 (zero-GT count) is **resolved to 6** by this audit (F-3); item 12 (AAIML deadline confirmation) remains undone and is schedule-critical. Day 4–7 (multi-task V4 with all 9 fixes): must add the DETACH wiring fix and a preset carrying `'detach_psr_fpn': False`, plus config-dump lines for `DETACH_PSR_FPN`/`PSR_HEAD_REPAIR` so no-ops are visible in logs. Week-2 and Week 3–4 plans: unchanged; the freeze discipline (hash checkpoint, re-run evals once, emit `results_frozen.json`) is the right control.

### §1.5 150 §§4–8 — final answers

- **§4 Honest story table:** verified with edits: PSR "0.78+ with V3 + real head repair" → "0.71–0.74 with V3; 0.78+ requires the attributed repair run (not yet run)"; detection same-backbone "near SOTA" is **pending**, not achieved.
- **§5 Beats/Near/Failures:** correct after (i) replacing "BEATS WACV 0.95" with "beats WACV 0.838/0.641, cross-architecture," (ii) adding the n=3 caveat to 0.6364, (iii) the never-predicted-class correction (F-3).
- **§6 Single most important question:** answered in §10 of this file.
- **§7 File locations:** audited in §1.1 above (per-path verdicts).
- **§8 Decisive test:** protocol is correct; branch logic updated in §10 to incorporate F-1/F-10 (a >0.78 V3 result must trigger the workstation check *before* attribution, and a ~0.70 result is a publishable negative, not a failure).
---

## §2. Document 150_SOTA_STATUS_V5 — Line-by-Line Verification

| Line / claim | Verdict |
|---|---|
| "Head Pose — 2/2 BEATS SOTA (first baseline)" | ⚠️ Internally contradictory label: there is no SOTA to beat (154 §2.4). Correct label: **first baseline**. Numbers/CIs ✅ verified (bootstrap_ci.json: 9.14 [7.74–10.87], 7.78 [6.89–8.81]). |
| "D1R single-task: 0.995 mAP50 (BEATS WACV 0.95)" | ⚠️ 0.995 ✅ verified (results.csv ep25 = 0.99484). "WACV 0.95" ❌ — uncited; use 0.838/0.641 (154 §1.1) with cross-architecture caveat. |
| "D3 multi-task: 0.00009 (impl bug, 4 fixes applied)" | ✅ verified (9.108e-05, d3_full_38k). Fixes in tree ✅ (sampler, gamma, assertions; anchor audit ruled anchors out). |
| "D4+D1R decisive: 0.6364 (decoder transfer verified)" | ⚠️ number ✅; add **n_videos=3** + "threshold-partial / marginal benefit" verdict string (F-5). "Verified" overstates. |
| "Per-comp optimal F1: 0.7018 (full 38k, honest)" | ✅ verified (0.70176). |
| "MonotonicDecoder F1: 0.0053 (saturated logits, will improve with repair)" | ✅ number consistent; "will improve" is a projection — no repair-fed decoder eval exists yet. |
| "PSR copy_prev F1: 0.9997 (model is 29.7% worse than persistence)" | ✅ arithmetic correct; must be co-reported with the prevalence-null deltas (F-11) or it reads as "no learning," which the null-delta table refutes. |
| "PSR head repair applied: LeakyReLU, activations **+384** on sequence frames" | ⚠️ Repair in tree ✅ (model.py:1604–1624). "+384" conflicts with "+4608" quoted in 150/151/152/156 (F-7); neither auditable remotely; also "applied" must be qualified per F-10 (running-process state unconfirmed). |
| Activity block (0.0236 / 0.2169 / **0.3810** / SOTA 0.622 / gate passed) | ✅ all verified (results.json; SOTA_STATUS). Note 0.3810's own baseline is clip-level 0.2666 (not 0.2217). |
| FiLM block (gamma 1.98, L2 27.7, std 0.002 — static 2×) | ✅ verified (film_gamma_beta.json per SOTA_STATUS row). |
| LOO-CV "+0.0148 ± 0.0158, CI includes zero; primary = 0.6788" | ⚠️ mean ✅; std is **0.0163** in loo_stratified.json (F-8). Conclusion unchanged. |
| Pose outlier block (model failure, GT clean, domain shift) | ✅ consistent with pose_outlier_analysis.md and the comparison memo. |
| "Implementation fixes applied (9 total)" list | ✅ all nine verified present in tree (code-level); cited hashes partially missing from this repo (F-4). |
| "What's in flight" (V3 epoch 24+, single-task det epoch 24+, MViT blocked, TCN+ViT blocked) | 🔒 workstation-only; note the internal inconsistency: this file says single-task det "epoch 24+" while 150/151 say "epoch 43+" — different snapshot times; harmless but shows these status lines go stale within hours. Also V3 must be re-labeled per F-1. |

---

## §3. Document 151_PER_HEAD_DEEP_ANALYSIS — Answers

**§1 Head Pose.** State ✅ verified except one number: §1.1 says per-recording median "8.94 / **5.82**" — the 5.82 is the forbidden 9-recording median-of-medians (140 §0 row 3); the all-16 median pair is **8.94 / 7.58** (pose comparison memo). §1.2 "Why it works" ✅ endorsed (spatial task, ImageNet features, simple head; ~25% Kendall share sufficient). §1.4 "beats uncited SOTA" ❌ — remove; first-baseline only.
**§1.5 Open questions answered:** (1) *Single-task pose ablation?* Not run; preset exists (`ablation_pose_only`, F-12); run as Week-2 filler; until then, zero multi-task-benefit language (the comparison memo already enforces this). (2) *Is 9.14° the limit?* No — best-recordings floor ≈ 6°; realistic single-task 7–8°; 5–7° optimistic. (3) *Does multi-task help or hurt pose?* Unknown; committed memo's verdict: "not a confirmed multi-task win; benefit marginal at best."

**§2 Detection.** §2.1 ladder ✅ verified (0.995 / 0.0004 / 0.358-biased / 0.00009 / 0.000→0.347@n16 / 0.000→0.6364@n3). §2.2 causes ✅ with the class-list caveat: never-predicted {1,13,16,19,23} ✅ (this file has it right; 150 Q1 had it wrong). "Class 12 default catch-all for 7 states" ✅ consistent with the confusion table (22→12, 21→12, 10→12 confusions). §2.3 four fixes ✅ in tree. §2.5 decisive-test branch logic ✅ endorsed; add the middle branch (0.1–0.5 = partial capability, cost reportable with caveat).

**§3 PSR.** §3.1 ✅ verified. §3.2 causes: GELU dead ✅ (mechanism now correctly located in `PSRHead.output_heads`); "+0.1 bias 1300× too small" ✅ (matches the model.py comment); "DETACH_PSR_FPN=True detaches gradient" ✅ true — **and still true in V3** (F-1), which corrects §3.3's claim that the V3 launch script fixed it. §3.5 "Expected F1 > 0.78 after 3–5 epochs" ❌ — superseded by 140 §-1/150: Kendall-only expectation is 0.71–0.74; >0.78 would itself demand re-attribution (workstation check first).

**§4 Activity.** ✅ verified throughout, including TCN-mean-pool 0.0723 (committed, so §4.2's "fails" is grounded) and the per-class recoveries. §4.5 fine-tune expectation 0.45–0.55: reasonable prior; flag the macro-mean caveat (F— see Q23: MViT probe non-zero classes 12 vs 22, macro mean 0.0545 vs 0.0580).

**§5 Cascade table.** ✅ with three edits: detection "Impl bug plus 8% GT batches" — add FP-density and IoU-0.234 as co-causes; PSR "Fix: V3 in flight" → "V3 = Kendall-only; attributed head-repair run still needed"; pose row ✅.

**§6 What we need to do.** Items 1–4 ✅ endorsed with V3 expectation reset (item 1's "F1 greater than 0.78" → 0.71–0.74) and the F-1 wiring fix added to any relaunch.

**§7 Honest verdict table.** Corrections: Detection "BEATS SOTA? YES (0.995 single-task)" — keep only with the cross-architecture label in the cell itself; PSR "NEAR SOTA? YES (0.78+ with V3 fix)" → "possible with attributed repair; V3 alone 0.71–0.74"; Pose "NEAR SOTA? YES (uncited 15 deg)" ❌ — delete the uncited comparison entirely.

**§8 Single-vs-multi verdict.** The four one-liners ✅ hold post-audit, with the sharpened statement: "Multi-task theory is untested here, not vindicated — what is proven is that three heads' failures have implementation/backbone causes that do not require multi-task interference as an explanation."

---

## §4. Document 152_IMPLEMENTATION_BUG_CATALOG — Answers

**§1 PSR head (GELU dead).** §1.1 bug ✅ (zero-fraction >0.97, pre-act −130, bias 1300× short, F1 vs copy-prev context). §1.2 fix code ✅ in tree at model.py:1604–1624 (the catalog's snippet cites lines 1597–1604 — drifted but same content; cited hashes not in this repo, F-4). §1.3 Agent-75 finding ✅ (DETACH_PSR_FPN=True default; all-11-heads RMS 0.00e+00; explains V1 failure). **§1.4 "The V3 Fix (28bf668c)" ❌ CORRECTED:** commit exists ✅ but the `export DETACH_PSR_FPN=False` line it adds is dead (F-1). The claimed causality "Result: post_gelu activations +4608" cannot be attributed to gradient flow through the FPN; the LeakyReLU/zero-bias re-init alone moves post-activations positive. §1.5 paths ✅ (repo-relative).

**§2 Detection.** §2.1 ✅ with the taxonomy fix (never-predicted {1,13,16,19,23}; zero-GT {1,2,3,14,15,23}). §2.2 all four fixes ✅ verified in tree; Fix 4's conclusion ("mapping correct; convergence issue") is the load-bearing sentence — it retires the mapping-bug hypothesis that 150 Q1 still repeats. §2.3 paths ✅.

**§3 Activity.** ✅ throughout; correctly labeled "NOT a code bug — a backbone type mismatch." The four built artifacts (video_backbones, video_backbone_multitask, activity_tcn, activity_tcn_vit) ✅ exist.

**§4 Pose index bug.** ✅ verified: eval-only; head_pose_diag.py fixed and deprecated; losses.py:951–952 always correct (checked directly). "3.5-month stale number" — plausible from file dates; not independently verifiable here.

**§5 FREEZE_BACKBONE.** ✅ verified: config.py:181–182 (`FREEZE_BACKBONE=True`, `BACKBONE_LR_MULT=0.01`), train.py param-group gating, `scripts/train_finetune_backbone.sh` exists.

**§6 The comprehensive fix list.** ✅ with two amendments: row 10 (V3 DETACH) is a **no-op as shipped** (F-1) — the table's "Effect: Gradient flows" is false; and four of the ten cited commits are not resolvable in this repository (F-4) — re-key before external citation.

**§7 The decisive question.** Answered: with all fixes *actually wired* (including F-1's one-liner), expectations are detection-MT 0.1–0.3 first relaunch (0.5–0.7 is the single-task expectation, not the multi-task one — the catalog's own table conflates them), PSR 0.74–0.80 with the attributed repair, activity 0.45–0.55 only after the 2-week fine-tune, pose unchanged. "Can we prove multi-task HELPS after all 9 fixes?" — see §10; the honest pre-registered criterion is the 153 §3.3 ratio test, and current evidence cannot anticipate its outcome.

---

## §5. Document 153_MULTI_TASK_DEBATE — Resolution

**Position A ("multi-task is fine, implementation was the killer").** Evidence audit: all five bullets verified as *facts*; their *inferential force* is: strong for PSR (head-local saturation + detach are task-agnostic defects), strong for detection (three measured causes, none requiring interference), decisive for activity (probes are single-task by construction — the failure reproduces without multi-task). Weakness: Position A's prediction "if V3 F1 > 0.78 multi-task is fine" was built on the un-wired repair (F-1/F-10) — the testable Kendall-only prediction is 0.71–0.74.

**Position B ("multi-task is hurting, even with fixes").** Evidence audit: the "8% positive gradient" and Kendall-collapse-in-theory points are real but now partially mitigated in tree (sampler; bounded/fixed Kendall). The "complex tasks fail, simple task works" pattern is genuinely suggestive but confounded: the complex tasks are exactly the ones with independent implementation defects. Position B's strongest surviving card: nobody has yet shown *any* head where multi-task ≥ single-task — and the pose memo explicitly withholds that claim.

**§3 Resolution matrix:** endorsed with corrected cells: PSR multi-task-with-fixes "0.78+?" → "0.71–0.74 (V3), 0.78+ (attributed repair, untested)"; detection multi-task-with-fixes "0.05–0.1?" is Position B's estimate and 152 §7's "0.5–0.7" is Position A's — print both as the pre-registered disagreement the V4 run settles. **§3.2:** 1 of 4 single-task baselines in flight; the other three are preset-ready (F-12) — the "blocked on GPU" framing understates readiness. **§3.3 ratio test** (≥0.9× = helps; <0.5× = hurts): endorsed as the pre-registered decision rule; add per-head application (the answer will differ by head).

**§4 The user's stance.** Adjudication on current evidence: the user is **right** that implementation is the dominant proximate cause for PSR and detection (measured, committed); **right-with-amendment** on activity (the wrong-backbone choice is itself an implementation-level decision, but no code fix rescues ImageNet features — the fix is architectural); **unproven** on "multi-task doesn't hurt" as a general claim — that requires the 2×2, and the only working head (pose) has an explicit no-claim memo. The document's own three-way split (2 impl / 1 backbone / 1 works) ✅ survives audit exactly.

**§5–§7.** Best-of-best path ✅ endorsed with the F-1 addition. §6 evidence gap ✅ correct: zero completed single-task numbers exist today — every "multi-task costs X" sentence is provisional until at least the detection denominator lands. §7 verdict ✅ as written, with the timeline note that "1–2 days (V3) / 3–4 days (single-task det)" are workstation ETAs unverifiable here.

---

## §6. Document 154_SOTA_COMPARISON — Answers

**§1 Papers.** WACV numbers (0.838/0.641/0.6525→0.6223 remap/B3 0.883/B1 0.779) and STORM (0.901/0.812/τ15.5s/B3-updated 0.891) — taken as accurately transcribed from the cited arXiv sources (2310.17323, 2510.12385); not re-derivable from this repo; the T3 verification (our 0.6223 = WACV remap) ✅ committed. §1.3 (arXiv 2408.11700 not comparable) ✅ reasonable.

**§2.1 Detection matrix.** ✅ verified per this audit (D1R 0.99484; D3 rows; conventions). The 0.573 rows: keep both flagged — ours is "derived, blocked ×3 (+ coincidence with WACV's synth-only 0.573)".
**§2.2 Activity matrix.** ✅ verified; the majority-baseline pair (0.2666 clip / 0.2217 per-frame) is correctly separated here — this file is the reference for that distinction.
**§2.3 PSR matrix.** ✅ verified; the "16.8% relative gap to STORM" decomposition (paradigm/architecture/implementation) is sound; add F-11's two-null clarification and the comp8 delta.
**§2.4 Pose matrix.** ✅ correct including the ~15° removal ruling; fix the per-recording-median row to 8.94/7.58 (not /5.82).

**§3 Fair-comparison tables.** §3.1 rows survive with edits: row 3 (MViT probe "61% of SOTA") ✅; row 6 relabel per F-11 ("persistence-dominated metric," not "worse than baseline" as a model-quality verdict); row 8 add n=3. §3.2 (comparisons that fail review) ✅ all nine correctly identified — this table is the document set's best defensive artifact; add row 10: "D4+D1R 0.6364 vs B3 0.883 — n=3 + post-hoc sweep."

**§4–§6 verdict tables.** ✅ with the standing corrections (cross-architecture caveat in-cell; V3 expectation reset; "BEATS SOTA (first baseline)" oxymoron → "first baseline").

**§7 "Can we beat SOTA on 2 heads + near SOTA on 2?"** Answer: **the defensible claim today is 1 beat (D1R, cross-architecture, same-protocol-0.641) + 2 first-baselines (pose fwd/up) + 1 near-SOTA-fraction (MViT probe at 61%)**; the realistic Jul-20 expectation adds the single-task ConvNeXt number (possibly near-SOTA vs 0.641) and V3's 0.71–0.74. The best case in §7 requires the attributed repair and the fine-tune — post-freeze. The worst case in §7 is already survivable because the two headline claims (pose baseline, PSR baseline-with-null-delta) are checkpoint-independent of the in-flight runs.

**§8 What the paper should claim.** ✅ endorsed as the master claim inventory, with counts updated: pathologies diagnosed 3 → **4** (add the F-1 env no-op), and "implementation bugs documented 3" likewise gains the wiring class.

**§9 Data integrity.** ❌ STALE in the document, ✅ in reality: the four "NEEDS COMMIT" rows are committed (F-6). Remaining genuinely-uncommitted evidence: the V3/single-task training logs and any post-V3 checkpoints (workstation).

**§10 References.** ✅ appropriate; add: this file (157) and the recovered 156 sections.
---

## §7. Document 155_FINAL_PAPER_NARRATIVE — Answers and Required Edits Before Submission

The narrative structure (pathology paper, three failure types, three lessons) is sound and endorsed. The draft as committed contains **ten defects that must be edited before submission**, each identified line-by-line:

1. **Abstract:** "BEATS the SOTA ceiling (mAP50 = 0.995 vs WACV 0.95)" — ❌ the ~0.95 is the uncited figure 140/154 ordered removed. Replace with "0.995 vs WACV 0.838 (annotated) / 0.641 (entire-video), cross-architecture."
2. **Abstract:** "PSR head GELU… (now fixed with LeakyReLU…)" — ⚠️ "fixed" must read "repair committed; validation run pending" until an attributed run produces an F1 (F-10).
3. **Training Configuration:** "Gradient flow to all heads is enabled via DETACH_PSR_FPN=False" — ❌ false for every run to date (F-1). Either wire the fix and re-run, or rewrite this paragraph to describe the detach-on state and move the no-op discovery into the pathology section (recommended — it strengthens the thesis).
4. **Results/Head Pose:** "BEATS uncited SOTA of approximately 15 degrees" — ❌ direct contradiction of 154 §2.4 ("Do not cite — source unverifiable"). Delete; keep first-baseline.
5. **Results/Head Pose:** per-recording median "8.94 / 5.82" — ❌ replace 5.82 with 7.58 (all-16 median of per-recording means; 5.82 is the forbidden 9-recording statistic).
6. **Results/Detection:** never-predicted classes "(1, 13, 16, 19, 23)" ✅ correct here — but ensure consistency with 150 (which has the wrong list, F-3) before reviewers diff the documents.
7. **Results/PSR:** "Expected F1 after repair is above 0.78" and Discussion "V3 PSR repair training is in flight (expected F1 > 0.78)" — ❌ reset to 0.71–0.74 (Kendall-only) with 0.78+ reserved for the attributed repair run.
8. **Discussion/Fix Path:** "All 9 implementation fixes are committed across 9 commits (e618d929a, …)" — ⚠️ four of the listed hashes do not exist in this repository (F-4); re-key to public-repo hashes or cite file+line instead.
9. **Conclusion/lessons:** lesson 1 lists three exhibits — add the fourth (DETACH env no-op, F-1), which is also the strongest because it was caught *by* the advocated method.
10. **Reproducibility:** cites `best.pth` SHA — add "checkpoint not distributed in-repo; SHA recorded for workstation verification," and fix `scripts/linear_probe_activity.py` (the actual committed probes live at `src/evaluation/activity_linear_probe*.py` / `activity_mvit_probe.py`).

**Answer to 155's implicit master question ("is this narrative submittable?"):** Yes after the ten edits, with the claim counts of 154 §8 and the §10 synthesis below as the abstract's spine. The three-lesson ending is the paper's best asset; the fourth exhibit makes it stronger.

---

## §8. Document 156_100_DEEP_QUESTIONS — All 100 Questions Answered

*Provenance note (F-2): §1, §4, §9 below were recovered from git commits `03ed3751d`, `8c9133a54`, `b08e01c76`; §2 (Q11–20) was never committed anywhere — the questions below for §2 are reconstructed from the section's title ("implementation critique") and its evident source material (file 152), and are labeled as such. All other sections are answered from the committed file.*

### §8.1 Multi-Task Theory Defense (Q1–10) — recovered

**Q1. Is multi-task learning a valid theoretical framework?** Yes — Caruana 1997 / Ruder 2017 and the modern practice record (Mask R-CNN-style multi-head vision systems, multi-task NLP pretraining) establish validity. The audit adds the discipline: validity is *conditional* on task affinity and implementation health; this project's evidence never impeaches the theory because every failure has a nearer cause (measured in §1.3 Q1/Q11/Q26). Verdict: theory valid; this project is not a test of it yet.

**Q2. Can multi-task help when tasks share representations?** Yes in general; here the four tasks share *low-level* visual features but diverge at the feature type that matters (spatial vs temporal). The committed probe evidence quantifies the divergence: ImageNet-spatial features fully serve pose (9.14°), partially serve PSR (null-delta positive on low-prevalence comps), and serve activity not at all (0.2169≈null). So sharing helps where representations actually overlap — which is precisely why the hybrid-backbone answer keeps recurring (Q72, Q80, Q97).

**Q3. What does Kendall uncertainty weighting guarantee?** Theoretically: gradient balancing via learned per-task uncertainty. Practically: no guarantee under extreme task-loss-scale imbalance — it can collapse; hence the in-tree mitigations `HP_PREC_CAP` and `KENDALL_FIXED_WEIGHTS` (config.py:94–115, env-gated, verified). V3 is (accidentally, F-1) the clean single-factor test of exactly this mitigation: 0.71–0.74 confirms weighting mattered modestly; flat ~0.70 is the publishable negative "weighting was not the binding constraint."

**Q4. Does the user's belief ("multi-task is fine") have literature support?** Yes, as a prior: typical published multi-task deltas are single-digit-% either direction, and catastrophic (0.00009-level) failures in the literature virtually always trace to defects, not to task interference. The audit's caution: literature support for the *prior* is not evidence for *this system* — the 2×2 remains the test.

**Q5. Is the V3 PSR repair evidence supporting multi-task theory?** ❌ CORRECTED as posed: the recovered text claims "the repair + DETACH_PSR_FPN=False fixes the implementation" — the DETACH half never executes (F-1), and the repair's presence in the running process is unconfirmed (F-10). What V3 *can* evidence: whether fixed Kendall weights matter (+0.01–0.03 band). "Activations +4608 alive" (🔒 log) shows the head can produce non-constant output — necessary, not sufficient, and explained by the re-init alone.

**Q6. Can the same ConvNeXt serve 4 different heads?** Three of four, with qualifications: pose yes (proven); detection plausibly (single-task run pending — the honest answer is "unknown until the denominator lands"); PSR partially (learned signal exists on low-prevalence comps; ceiling unknown until an attributed repair run). Activity: **no** — proven by three probes (0.2169, 0.0723, vs MViT 0.3810). The recovered text's "architecture is the bottleneck, not the backbone sharing" is exactly backwards for activity: the backbone (pretraining domain) is the bottleneck; sharing is incidental.

**Q7. Does the cascade table prove multi-task theory is wrong?** No — endorsed. Each broken head has a specific, measured, task-local cause (GELU saturation; empty-frame flooding + confusion + IoU; backbone domain). The cascade table is evidence about implementation, silent about theory. (The cited `multi_task_cascade/cascade_table.md` path is not in this repo's checkpoints dir; the equivalent content is 151 §5 — minor citation fix.)

**Q8. Can multi-task work with the right architecture?** The four listed conditions (Kinetics backbone for temporal tasks, temporal heads, bounded Kendall, imbalance-aware losses) are each individually evidenced in this project's record; their conjunction is untested. Honest answer: plausibly yes; the video_backbone_multitask.py design (53.8M/19.3M) is the concrete proposal; it is post-freeze work.

**Q9. What's the strongest evidence FOR multi-task being correct?** Ranked: (1) pose works *inside* the 4-task system — existence proof that the shared trunk carries at least one head to a strong result; (2) PSR's prevalence-null deltas (+0.097/+0.093/+0.053) — a second head learns real signal in-system; (3) D1R 0.995 proves the data/split supports near-perfect detection, so the multi-task failure is not a data ceiling. The recovered text's "linear probe shows backbone has signal" must be dropped — the ConvNeXt probe is a **null** result (the "BACKBONE HAS SIGNAL" claim was formally retracted in 140).

**Q10. Should the paper claim multi-task helps?** No, under every currently-possible outcome — the conditionals in the recovered text (V3>0.78, det>0.5, MViT>0.45) test *recoverability*, not *help*. "Help" requires multi-task > single-task same-everything, and zero such comparisons exist (F-12 notes even pose withholds it). The paper claims: measured costs, diagnosed causes, fix paths — and explicitly labels "does multi-task help?" as the open question the 2×2 answers post-freeze.

### §8.2 The Implementation Critique (Q11–20) — section lost (F-2); reconstructed from file 152's material and answered

*The ten questions below are this audit's faithful reconstruction of what a section titled "implementation critique" sitting between "theory defense" and "backbone analysis" must cover; each is answered with committed evidence.*

**Q11 (rec.). How many distinct implementation defects have been verified, and what taxonomy do they fall into?** Nine fixed + one meta-defect, in four classes: **activation/init pathology** (PSR GELU saturation; +0.1 bias 1300× short; fixed via LeakyReLU + small-normal + zero bias, model.py:1604–1624); **data/sampling pathology** (8%-positive batches → GuaranteedGTBatchSampler; DET_GAMMA_NEG 1.5→2.0); **measurement pathology** (up-vector [3:6] eval bug; full-eval v2 index corrections; temporal-probe bare-except); **wiring/observability pathology** (PSR_HEAD_REPAIR consumed only by dead code; DETACH_PSR_FPN env no-op, F-1; config dump omitting both flags). The fourth class is the paper's thesis.

**Q12 (rec.). Which defect cost the most performance?** By head: detection — the sampler/FP-density pair (bounded by 0.00009 → single-task pending); PSR — the head saturation (bounded by the zero-RMS gradient evidence across all 11 sub-heads); activity — not a code defect (backbone domain); pose — none (eval-only bug cost 3.5 months of *reporting*, zero training loss). Portfolio answer: the *observability* defects cost the most calendar time, because each silent no-op consumed a full training-and-evaluation cycle before detection.

**Q13 (rec.). Why did four independent defects survive so long?** Common mechanism, stated by 140 §-1 and now with four exhibits: all four were invisible to the loss curve. GELU-dead heads still produce a decreasing PSR loss (on the easy prevalence signal); an eval-index bug changes no training metric; an unconsumed env var changes nothing at all; dead code trains nothing. None trip an exception. The missing layer is per-path runtime verification: assert-at-startup that every env toggle is consumed, log effective config (train.py's dump prints KENDALL_FIXED_WEIGHTS but not PSR_HEAD_REPAIR/DETACH_PSR_FPN — the tell), activation-statistics monitors, and per-task eval sanity bounds.

**Q14 (rec.). Was the +0.1 bias guard a reasonable engineering decision at the time?** Yes-but: it encoded the right diagnosis (GELU zero-collapse risk, named in the `[AUDIT]` comment) with an unvalidated magnitude — pre-activations at −130 needed a ~+130 shift, 1300× the guard. The lesson is not "the guard was wrong" but "guards need measurements": a one-batch activation histogram at init would have sized it correctly. This is Q13's thesis applied prospectively.

**Q15 (rec.). Is the GT-balanced sampler a principled fix or a hack?** Principled for optimization (guarantees positive detection gradient per batch), with a distribution-shift cost that must be tracked (train-time GT prior ≫ eval-time 8.1% — see §1.3 Q9). The alternative (loss reweighting on natural sampling) is cleaner statistically but was already partially in place (focal loss) and insufficient. Verdict: right fix now; report the train/eval prior gap in the paper's limitations.

**Q16 (rec.). Did mixed precision (bf16) contribute to any pathology?** No committed evidence implicates bf16 in the four verified defects; the wrapper's MIXED_PRECISION override is itself cleanly implemented (patches before train.py import, re-applies post-preset). The NaN-checkpoint-selection pathology cited among the exhibits (140/154 §8) is checkpoint-selection logic, not AMP. Keep AMP_DTYPE in the config dump for auditability.

**Q17 (rec.). Are the detection anchors and class mapping exonerated?** Yes, by commits in this repo: anchor audit `10d5ab596` (k-means on 14,122 GT boxes; >99% coverage at IoU 0.5; "not the root cause") and class-index assertions `a0ffb9aa8` ("mapping correct; 5 never-predicted classes is a training convergence issue"). These two negative results matter: they cut the detection hypothesis space to sampling + loss shaping + head capacity.

**Q18 (rec.). Is the PSR loss design itself sound?** The Gaussian-smeared transition objective is live (`build_transition_targets(sigma=3)`, losses.py:1436–1454, applied on sequence batches, skipped on per-frame batches — verified live by 140). Open design questions inherited from 135 (sigma choice, focal-vs-BCE per component, sequence-batch fraction) are untested, not defects. They become relevant only after the head repair run establishes the new baseline.

**Q19 (rec.). What single cheap instrument would have caught all four wiring defects?** An "effective-config + live-module manifest" printed at step 0: every config flag with its post-preset value (catches F-1), every env var consumed vs ignored (catches PSR_HEAD_REPAIR), every nn.Module in the forward path with parameter counts (catches dead PSRTransitionPredictor), plus per-head activation/gradient RMS at steps {0, 100, 1000} (catches GELU death within minutes instead of epochs). Estimated cost: <100 lines. This belongs in the paper as the constructive recommendation.

**Q20 (rec.). After all fixes, what residual implementation risk remains?** Ranked: (1) DETACH_PSR_FPN wiring — still unfixed (F-1); (2) checkpoint-resume overwriting re-initialized PSR heads (no confirmed resume-exclusion for `output_heads` — §1.3 Q19); (3) transformer output-variance collapse (diagnostic pending); (4) V4 launch inheriting stage_rf4's detach-True preset silently; (5) eval-protocol drift between the single-task and multi-task detection evals (frozen-results discipline mitigates). Items 1, 2, 4 are one-line fixes; do them before V4.

### §8.3 Backbone Architecture Analysis (Q21–30) — committed

**Q21. Is ConvNeXt-Tiny the right backbone for 4-task multi-task?** Per-task verdicts verified: pose YES; detection UNKNOWN-pending (the committed text's "maybe" is right; note its own aside "linear probe 0.2169" mislabels an *activity* probe as PSR evidence — the PSR head does get positive null-deltas from these features); activity NO (proven). Composite: ConvNeXt-Tiny is defensible for 3 of 4 heads and indefensible for activity; hence hybrid (Q24/Q80).

**Q22. What does MViTv2-S give us that ConvNeXt doesn't?** Kinetics-400 action pretraining ⇒ features that linearly separate assembly actions: +0.164 top-1 under identical probes, 11 zero-class recoveries (mvit_per_class). Also spatiotemporal attention (16-frame clips) vs per-frame GAP — the probe's clip-level protocol is part of the gain. All committed.

**Q23. Should we replace ConvNeXt with MViTv2-S?** For activity: yes (gate passed, 0.3810 > 0.30). For the whole system: no — replace *for the temporal head(s)* via the hybrid design; pose/detection have no evidence of needing it and MViT multi-task convergence is the stated risk. Cost (2 GPU-weeks) and expected value (0.45–0.55) per the committed plan.

**Q24. Can we use a hybrid backbone (ConvNeXt + MViTv2-S)?** Yes — designed and committed (`video_backbone_multitask.py`, 53.8M/19.3M trainable). The recovered text's "share early layers, split later" overstates what's possible across heterogeneous architectures — the committed design is two backbones with task routing, which is the honest description. Untrained; post-freeze.

**Q25. Does the backbone need Kinetics pretraining for activity?** Yes — the cleanest single finding in the project: identical linear heads, frozen features, ImageNet 0.2169≈null vs Kinetics 0.3810. Add the temporal-probe corroboration (0.0723 < baseline: even temporal aggregation cannot rescue ImageNet features).

**Q26. Is per-frame architecture the problem for activity?** Secondary. The committed text is right that per-frame MLP can't model dynamics, but the controlled evidence shows the backbone dominates: temporal aggregation on ConvNeXt *fails* (0.0723) while a *non-temporal linear probe* on MViT succeeds (0.3810). "TCN on MViTv2-S: pending" ✅ still true.

**Q27. What's the right architecture for 4-task multi-task?** Endorsed as committed: hybrid backbone; per-task heads (FPN detection, linear pose regression, TCN/attention activity, causal-transformer PSR); bounded/fixed Kendall; imbalance-aware losses. Every element individually exists in tree; the conjunction is the post-freeze experiment.

**Q28. Can we use ConvNeXt for 3 heads + MViTv2-S for activity?** Yes — this is the minimal hybrid and the recommended configuration (matches Q78's "minimal architecture change"). Cost: one extra forward pass per clip for the activity path; benefit: no risk to the three working/fixable heads.

**Q29. How long does MViTv2-S fine-tuning take?** 2 weeks per run as committed; the "16 weeks total for complete ablation" arithmetic is right but collides with the Jul 20 freeze — the committed "achievable in 1 quarter" is the *post-submission* program. For the paper: frozen-probe number only, with fine-tuning as stated future work (or a camera-ready update if the venue permits).

**Q30. What's the best backbone choice for IndustReal?** As committed, with sharpened evidence tags: MViTv2-S for activity (proven direction), ConvNeXt for pose (proven), detection either (unknown — D1R proves YOLOv8m works; ConvNeXt pending), PSR ConvNeXt-adequate-pending-repair (null-deltas positive). Hybrid is "best of both" by construction; 75–100h per architecture choice is the committed budget figure.

### §8.4 Detection Head Debate (Q31–40) — recovered

**Q31. Why does D3 get 0.00009 mAP?** Answered with the full measured decomposition in §1.3 Q1 (flooding 96.2% FP mass, 46% classification correctness, IoU 0.234, 1.4% true detections). The recovered text's four bullets are all verified; "Agent-55 root cause analysis" = `detection_root_cause/analysis.md` ✅ committed.

**Q32. Is detection broken by multi-task or implementation?** Implementation-dominant on current evidence; the controlled answer arrives with the single-task run. Note the recovered text correctly lists never-predicted {1,13,16,19,23} — 150 Q1's divergent list is the error (F-3).

**Q33. Will the 4 detection fixes make D3 work?** Necessary-not-sufficient; expected recovery band 0.1–0.3 for the first V4 relaunch (the recovered "0.1–0.5" upper half assumes confusion/IoU also improve from increased positive exposure — possible, unproven). Full answer §1.3 Q3.

**Q34. Can multi-task detection beat single-task?** No current evidence path to "beat"; the achievable target is the ≥0.9× ratio (153 §3.3). The recovered text's structural point stands: 91.9% empty frames afflicts both regimes equally — it explains hardness, not the multi-task *gap*.

**Q35. Is D1R the right comparison for multi-task?** No — cross-architecture ceiling only (§1.3 Q4). The fair comparison is single-task ConvNeXt vs multi-task ConvNeXt, both post-fixes; the former is in flight.

**Q36. What does D4+D1R = 0.6364 tell us?** Detection density binds the decoder — direction verified; magnitude caveated (n=3 videos, post-hoc global sweep, verdict-string "marginal benefit"; F-5). The 83% relative improvement over 0.3466 mixes n=3 and n=16 evaluations — recompute on matched video sets before publishing the percentage.

**Q37. Why does D4 default = 0.000?** Threshold regime mismatch, verified: Q48-era hysteresis (hi 0.5/lo 0.3/sustain 3) assumes dense confident detections; pretrained YOLOv8m under our pipeline fires <1%/frame; re-tuned (0.3/0.1/2) → 0.3466@n16. Committed in `d4_retuned/`.

**Q38. Should we report detection per-class?** Yes — and the zero-GT list here ({1,2,3,14,15,23}) is the correct one, now confirmed by the d3 JSON keys (F-3). Report: per-class AP (18 present), prediction-volume vs GT-volume table (the 2,559:1 / 6,171:1 ratios are the most legible evidence of collapse), and the never-predicted trio {13,16,19} called out.

**Q39. What's the true multi-task detection ceiling?** Unknown; the recovered estimate (0.05–0.15) is Position-B-flavored, 152 §7's 0.5–0.7 is Position-A-flavored — print both as the pre-registered disagreement (§5). The V4 run adjudicates.

**Q40. Should we cut detection from the paper?** No. The recovered recommendation ("cut multi-task detection, report D1R") is superseded by the pathology framing: the 0.00009-with-causes is *content*, not embarrassment, and cutting it while keeping D1R would be exactly the cherry-picking 154 §3.2 warns about. Report the ladder: 0.00009 → fixes → V4 number → single-task denominator → D1R ceiling.

### §8.5 PSR Head Debate (Q41–50) — committed

**Q41. Is the PSR head broken by implementation or architecture?** Implementation, three layers deep, all verified: GELU saturation (heads), +0.1 bias magnitude error, DETACH_PSR_FPN isolation (config) — plus the candidate upstream variance collapse (diagnostic pending). The committed text's "V3 fix: … + DETACH=False" ❌ carries the F-1 error; "post_gelu −1.0/−1.4 → +4608" 🔒 log-only (and F-7's +384 conflict).

**Q42. What does V3 F1 = 0.78+ mean if achieved?** Re-answered post-F-1: it would mean *either* the running process loaded the repaired heads (check first — F-10) *or* Kendall weighting alone was worth ≥+0.08 (would overturn the +0.01–0.03 estimate). It cannot cleanly mean "the repair works" until attribution is settled. Full branch logic §1.3 Q12.

**Q43. Why is our PSR F1 ≈ null_copy_prev F1?** It isn't "≈" — it's 0.30 *below* (0.7018 vs 0.9997), and the correct reading is F-11's: copy-prev is a near-unbeatable persistence ceiling under extreme temporal autocorrelation, POS is structurally inflated, and learning is measured by the prevalence-null deltas (+0.097/+0.093/+0.053). The committed text's "this suggests the head is broken" is half-right: broken *for transitions*, provably non-trivial *for states*.

**Q44. What's the gap between MonotonicDecoder F1 and our head F1?** Verified ladder: decoder-on-our-logits 0.0053 (full 38k — the honest same-data comparison, and it indicts the logits, not the decoder); decoder 2-rec 0.7893 (artifact, retired); head 0.7018/0.6788. The committed text's conclusion (keep the head, fix the implementation) ✅ endorsed.

**Q45. Why is the procedure-order constraint the bottleneck?** Verified from the committed text itself: oracle-with-order 0.5947 vs relaxed 0.8807 (32-point gap); comp4 never placed in 10/16 recordings. This is the quantification 150 Q17 said didn't exist — the two documents disagree and **156 is right** (see §9 ledger). Data-driven ordering (or per-recording order inference) is the design implication.

**Q46. Should we replace the PSR head with the MonotonicDecoder?** No — verified: head 0.7018 ≫ decoder 0.0053 on identical data; the decoder's better-looking numbers come from small samples or externally-supplied dense detections. Keep head; decoder as optional post-processing reported separately.

**Q47. What's the right PSR architecture for assembly?** Endorsed as committed (LeakyReLU + small-normal init; detach off — *once actually wired*, F-1; causal temporal module — already present as the 3-layer causal transformer, model.py:1579–1589; multi-scale dilations if TCN; data-driven ordering per Q45). One addition: the variance diagnostic decides whether the transformer itself needs re-init/LR changes.

**Q48. Is multi-task PSR worth the implementation cost?** The committed decision tree (V3 vs single-task PSR comparison) stands with reset expectations (V3 0.71–0.74). `ablation_psr_only` preset exists (F-12) — the single-task arm costs ~2–3 GPU-days and completes the head's 2×2 row; recommended Week-2 filler behind the detection denominator.

**Q49. What does LOO-CV +0.0148 ± 0.0158 mean?** Verified with the F-8 correction (±0.0163): per-component threshold optimization does not transfer significantly across recordings (CI includes zero); therefore global-0.10 (0.6788) is the honest primary and per-comp-optimal (0.7018) is reported as the calibrated upper variant. All 16 recordings are val-only (loo_stratified.json: train list empty) — the train/val-contamination worry from 135 Q20 is formally closed by this artifact.

**Q50. What should we report for PSR in the paper?** The five-row set, all verified: global-0.10 0.6788 (primary); per-comp-optimal 0.7018 [CI 0.6436–0.7321]; prevalence-null delta table (all 11 comps incl. comp8, F-11); copy-prev 0.9997 + POS-null table as paradigm caveats; decoder 0.0053 as diagnostic. Plus V3's number labeled Kendall-only when it lands. STORM/B3 comparisons only in a paradigm-difference table, never same-row.
### §8.6 Activity Head Debate (Q51–60) — committed

**Q51. Is activity broken by implementation or backbone?** Backbone-dominant — verified by the probe triangle (0.2169≈null / 0.0723<null on ImageNet features; 0.3810 on Kinetics features). The implementation co-factor is real but historical: `ACTIVITY_GRAD_BLEND_RATIO` was 0.05 (starving activity gradients) and is now 1.00 in tree (config.py:1035) — even at 1.0, ImageNet features cannot carry the task.

**Q52. What does MViTv2-S 0.3810 tell us?** Verified: Kinetics features linearly separate assembly actions at 61% of protocol-matched SOTA with zero fine-tuning (+0.114 over the clip-level 0.2666 majority). It localizes the failure to pretraining domain, and it sets the floor for the fine-tuning investment decision (gate 0.30 passed).

**Q53. Why does the MViT probe work but multi-task 0.0236 fail?** Because the two differ in *backbone*, not in multi-task-ness — the probe is single-task by construction, but the ConvNeXt *probe* is also single-task and fails identically (0.2169≈null). The controlled pair (same probe, two backbones) isolates the cause cleanly. The committed text's "multi-task + Kinetics = best of both worlds" is a projection, not a result — untested.

**Q54. Which classes benefit most from video features?** Verified from `mvit_per_class/comparison.md`: check_instruction 0→0.8771 (529 val clips), tighten_nut 0→0.7149 (235), plug_objects 0→0.3558 (104), take_objects 0→0.2989, align_objects 0→0.1547; 11 zero→nonzero transitions total. Pattern: high-support, visually-extended activities recover first — consistent with a linear probe's sample-efficiency limits.

**Q55. Why do some classes go to zero with video features?** Verified: 19 ConvNeXt-nonzero classes hit zero under the MViT probe (pull_pin_middle 62.1%→0, take_pin_long 50.4%→0, put_wheel 33.2%→0); MViT probe has 12 nonzero classes vs ConvNeXt's 22, and a *lower* macro per-class mean (0.0545 vs 0.0580). Two readings: (a) committed text's — rare classes lack probe training samples, fine-tuning will recover; (b) audit's addition — some short/handheld actions may be encoded in ConvNeXt's *spatial* signature (which object is held) and genuinely lost in clip-level pooling. Report both; the fine-tune adjudicates.

**Q56. Is per-frame MLP the problem for activity?** Secondary to backbone — answered with controls at §8.3 Q26. The committed "TCN mean-pool on ConvNeXt: 0.0723 (fails)" is verified-committed evidence; "TCN on MViTv2-S: pending" correct.

**Q57. Should we use TCN+ViT for activity?** Yes, gated exactly as committed: ACT-1 gate (probe>0.30) passed by MViT only ⇒ TCN+ViT justified *on MViT features* and closed on ConvNeXt features. Sequencing per §1.3 Q22: fine-tune first, temporal head second.

**Q58. Is 2-week MViTv2-S fine-tuning worth it?** Yes by expected value (0.3810→0.45–0.55 toward SOTA 0.6223; script `train_mvit_finetune.sh` committed), but it cannot fit before Jul 20 — so "worth it" resolves to: launch when GPU frees, paper ships with the frozen-probe number and the fine-tune as declared future work.

**Q59. Should multi-task activity use MViTv2-S?** Eventually; single-task anchor first (§1.3 Q28). The committed 5–10% multi-task-drop projection is a prior with no in-project evidence — label it as such.

**Q60. What's the right framing for activity in the paper?** Verified-endorsed: 0.0236 = pathology exhibit (never SOTA-compared); 0.3810 = first video-backbone probe baseline (reportable, protocol-labeled); 0.45–0.55 = projection (labeled); the arc "wrong backbone → diagnosed by probes → confirmed by backbone swap" is the section's spine. Matches 140 Decision 4.

### §8.7 Pose Head + Cross-Task Debate (Q61–70) — committed

**Q61. Why does pose work when other heads fail?** Verified reasoning: spatial task ⟂ ImageNet features; simplest head (GAP→linear, no saturating bottleneck); per-frame regression matches the label structure; ~25% Kendall share sufficient. Add the audit's sharpening: pose also has the *densest* supervision (a label every frame, no imbalance) — supervision density, not just feature match, separates it from detection (8.1% frames) and PSR transitions (rare events).

**Q62. Is 9.14° a real "first baseline" claim?** Yes-gated (literature-search artifact still uncommitted); "BEATS uncited SOTA ~15°" ❌ must go (154 §2.4). Full answer §1.3 Q31.

**Q63. What does the 3.5-month index bug tell us?** Verified (eval-only; 4 scripts fixed; training always correct — losses.py:951–952 checked). One correction: the committed text repeats "per-rec median 5.82° is more honest" — ❌ 5.82 is the forbidden 9-recording statistic; 7.58 is the all-16 median (F— §1.3 Q32). Lesson: per-task sanity bounds on eval outputs.

**Q64. Is multi-task helping or hurting pose?** Unknown — and the committed memo (pose_multitask_vs_singletask) formalizes exactly the recovered text's branch logic and then withholds the claim. Answer stands: run `ablation_pose_only` or say nothing about transfer.

**Q65. What's the 14_assy_0_1 outlier?** Verified: 17.05/12.32 vs next-worst 11.49; model failure, GT clean, domain-shift hypothesis. One typo in the committed text: "report both with-outlier (7.39° fwd) and without (9.14° fwd)" has the numbers backwards — with-outlier is 9.14, without is 8.46 (forward); 7.39 is *up* without outlier. Corrected here.

**Q66. Can we beat 9.14° with single-task pose?** Possibly, to ~7–8° (floor analysis §1.3 Q35); the "100% vs 25% gradient" argument is directionally fine but ignores the ~6° GT-noise floor. 5–7° is optimistic.

**Q67. Is Kalman smoothing worth the 1.5–2.7%?** As an accuracy gain: marginal. As evidence: yes — it certifies temporal smoothness of raw predictions (worth one line), and its per-recording pattern (helps most where error is highest) is diagnostic. Real headroom is upstream (features/architecture), as committed.

**Q68. What's the right pose architecture for SOTA-comparable?** Endorsed: current 9-DoF regression + single-task gradient; optional SO(3)-aware loss and RTS smoothing already explored. Position reporting stays off until units are verified vs HoloLens export (154 §2.4). "Single-task expected 5–7°" → 7–8° realistic.

**Q69. Should we cut pose from the paper?** No — verified consensus across all documents; it carries two of the paper's headline claims. Keep with first-baseline framing, weighted-mean primary, and zero multi-task-benefit language (memo-enforced).

**Q70. What's the cross-task learning transfer opportunity?** The committed answer ("these don't share representations; one shared backbone cannot serve all 4") is directionally supported for activity but overclaims for detection/PSR/pose, which *do* share usable spatial features (pose works; PSR null-deltas positive). Refined answer: transfer opportunity exists among spatial tasks (pose↔detection↔PSR-state); the temporal task needs its own feature stream — hence hybrid, not four separate models.

### §8.8 Architecture Options (Q71–80) — committed

**Q71. What options are on the table?** Verified list (A ConvNeXt-Tiny 28M / B MViTv2-S 36M / C hybrid 64M / D VideoMAE 86M / E TimeSformer 121M / F ConvNeXt-V2 198M). Param figures are catalog values; only A–C have in-project evidence.
**Q72. Best for 4-head multi-task?** C (hybrid) — endorsed; evidence: A proven for pose + failed for activity; B proven for activity (probe) + unproven elsewhere. D/E excluded on cost, F on redundancy (still ImageNet-domain).
**Q73. Best for single-task detection?** Honest answer: unknown pending the in-flight run; D1R proves the YOLO family suffices — so the *practical* best is "YOLOv8m if detection is standalone; ConvNeXt if it must share." The committed "C or F" recommendation has no supporting measurement — flag as speculation.
**Q74. Best for single-task activity?** B (MViTv2-S fine-tuned) — verified-cheapest with in-project evidence (0.3810 probe). Endorsed.
**Q75. Best for single-task PSR?** A-after-repair (endorsed) — PSR's evidence (positive null-deltas on ConvNeXt features) says the features suffice; the head was the defect. B/C only if the repaired ceiling disappoints.
**Q76. Best for single-task pose?** A — verified sufficient (9.14° with 25% gradient share); anything larger is unmotivated spend.
**Q77. Time/cost constraints?** Verified arithmetic (2wk/change; 8+8wk for full ablation; 1 quarter) — with the standing caveat that this is the post-submission program (Jul 20 freeze fits only the in-flight runs + cheap evals; §8.3 Q29).
**Q78. Minimal architecture change for SOTA-comparable?** MViT single-task for activity only — endorsed, matches Q28; 2 weeks; zero risk to other heads.
**Q79. Maximal architecture for SOTA-beating?** The committed 3-backbone maximalist plan (VideoMAE + TimeSformer + ConvNeXt-V2) is unsupported by any in-project measurement and would forfeit the "one system" story entirely — deprioritize; the hybrid already captures most expected value.
**Q80. Right architecture choice for the user?** Hybrid ConvNeXt+MViTv2-S (`video_backbone_multitask.py`, committed design) — endorsed as the standing recommendation, with the discipline that its first training run is post-freeze and its projections stay labeled.

### §8.9 Single-Task vs Multi-Task (Q81–90) — recovered

**Q81. What single-task baselines do we have?** Recovered text verified and improved: detection in flight (🔒); activity/PSR/pose **preset-ready in tree** (`ablation_act_only`/`ablation_psr_only`/`ablation_pose_only`, F-12) — "script ready/not started" understates; each is a one-line launch when GPU frees. Status: 1 of 4 running, 3 of 4 config-complete, 0 of 4 completed.
**Q82. Single vs multi for pose?** Branch logic endorsed; expected single-task 7–8° (not 5–7, §8.7 Q66); memo already strips benefit language either way.
**Q83. Single vs multi for PSR?** Reset per F-1/F-10: V3 expectation 0.71–0.74 (not >0.78); single-task PSR unrun; the comparison is meaningful only after the attributed repair exists on the multi-task side, else it compares a repaired single-task against an unrepaired multi-task.
**Q84. Single vs multi for activity?** Verified numbers; note the comparison spans backbones (ConvNeXt multi-task 0.0236 vs MViT single-task 0.3810), so it measures backbone + task-count jointly — only the future MViT-multi-task run isolates the multi-task term. The recovered "loses 5–10%" is the unlabeled prior again.
**Q85. Single vs multi for detection?** Verified branch logic; this is the decisive pair of the whole program (§10).
**Q86. Can the 2×2 ablation prove multi-task helps?** Yes in design; endorse the recovered thresholds (≥0.9× helps-slightly / ≥1.1× strongly / else hurts) as pre-registered, applied per-head, with identical eval protocol (frozen results discipline).
**Q87. Cost of 4 single-task baselines?** Verified: 8–12 days sequential — the recovered per-task estimates stand; presets exist so setup cost ≈ 0.
**Q88. Can we run them in parallel?** Workstation constraint (2 GPUs, memory) 🔒 — accept the recovered sequential answer; note the 3060 can carry the cheap CPU/1-GPU-hour probes concurrently (temporal probe precedent).
**Q89. Right way to compare?** Endorsed as committed: same backbone/data/aug/optimizer/schedule/epochs, task set as the only variable — this is exactly what the A1–A4 presets encode ("arch+hparams == stage_rf4"), which is why F-12 matters: the controlled comparison was designed in already.
**Q90. What if single-task wins on all 4?** Then the paper's framing survives unchanged (measurement paper reporting the cost), the *theory* discussion cites task-affinity literature rather than declaring multi-task dead, and the hybrid direction (per-task feature streams under one system) becomes the constructive conclusion — as the recovered text anticipates.

### §8.10 Best-of-Best Path Forward (Q91–100) — committed

**Q91. Immediate next step (1–2 days)?** Get V3's F1 — endorsed with corrected branch labels: 0.71–0.74 = Kendall fix works; ~0.70 flat = weighting not binding (publishable negative); >0.78 = run the workstation attribution check before celebrating (F-10). Also immediate and free: the F-1 one-line wiring fix + config-dump lines, and the workstation no-op check itself.
**Q92. Medium-term (1–2 weeks)?** Endorsed: single-task detection completes → denominator; launch MViT fine-tune when GPU frees; add: launch `ablation_psr_only`/`ablation_pose_only` as fillers (F-12), and re-eval multi-task detection full-set post-fixes so the V4 comparison has both sides.
**Q93. Long-term (1 quarter)?** Endorsed as committed (4 single-task + 4 multi-task conditions, 2×2, paper from data). This is the post-submission program; Jul 20 ships the measurement paper.
**Q94. Headline result for the paper?** Corrected version of the committed list: 1 cross-architecture beat (D1R 0.995 vs 0.641 same-protocol), 2 first baselines (pose fwd/up), 1 first per-frame PSR baseline with null-delta-proven signal, 1 probe breakthrough (0.3810), 4 pathology exhibits, 9 fixes with public commits, denominators pending. "2 BEATS + 2 NEAR" overstates today's evidence (near-SOTA rows are projections).
**Q95. What's the user hoping to prove?** Verified against the record: implementation is the dominant cause for PSR + detection (supported), backbone for activity (supported), "multi-task can beat all SOTA" (open — requires the 2×2 and the fine-tune). The honest restatement: the user's hypothesis has survived every test so far *because* every failure found a non-multi-task cause; it has not yet been affirmatively confirmed anywhere.
**Q96. Can multi-task beat SOTA on all 4 heads?** Not on current evidence, and not by Jul 20. Best defensible trajectory: near-SOTA on detection (post-V4) and activity (post-fine-tune), first-baselines on pose/PSR, with SOTA-beating plausible only for detection-single-task (already done, cross-architecture). The committed "1 BEATS, 3 NEAR" best case is fair *as a best case*, all three NEARs being projections.
**Q97. Right architecture for the user?** Hybrid (Q80) — consistent across all documents; committed design; post-freeze training.
**Q98. Right training strategy?** Endorsed: Phase 1 single-task baselines (definitive), Phase 2 multi-task V4 with all fixes **including F-1's wiring** + MViT hybrid, Phase 3 2×2, Phase 4 writing. One resequencing note: Phase-4 writing starts *now* in parallel (the pathology sections need no pending numbers).
**Q99. Most important question for the user?** As committed ("can the right architecture + all 9 fixes make multi-task beat or near SOTA?") with the answer's honest form: YES for 2 heads today (pose first-baseline; D1R ceiling), CONDITIONAL for PSR/activity on the attributed repair and the fine-tune — and the *decision-relevant* answers arrive from the two in-flight runs within days (§10).
**Q100. Final synthesis for AAIML?** Endorsed with the corrections embedded throughout this file: the user's implementation hypothesis is supported for 2 heads, amended for 1 (backbone), untested for the general claim; 9 fixes committed (one no-op to re-wire, F-1); V3 is the Kendall ablation; MViT fine-tune is next; the paper is "What Four Tasks Cost One Backbone"; the contribution is pathology + fix path + first baselines; "beat or near SOTA on 4 heads" is the post-freeze target, not the submission claim.

### §8.11 The Final Verdict (156 §11) — audited

The committed verdict ("the user is right; with 9 fixes: pose 9.14 BEATS, D1R 0.995 BEATS, PSR →0.78+ with V3, activity →0.45+ with MViT") is **directionally endorsed and numerically corrected**: pose = first baseline (nothing to beat); D1R = cross-architecture beat; PSR = 0.71–0.74 via V3, 0.78+ only via the attributed repair run; activity = 0.3810 today, 0.45+ projected. "The 100 questions are answered by the in-flight trainings" — more precisely: ~10 of them are (the decisive ones); the rest are answered by the committed evidence as done in this file.

---

## §9. Cross-Document Contradiction Ledger

Every internal inconsistency found across 150–156, with the resolution this audit adopts:

| # | Contradiction | Documents | Resolution |
|---|---|---|---|
| 1 | Never-predicted classes {1,2,3,14,23} vs {1,13,16,19,23} | 150 Q1 vs 151/152/155/156-Q32 | **{1,13,16,19,23}** (root-cause analysis proves 2,3 fire); zero-GT = {1,2,3,14,15,23} (F-3) |
| 2 | "Class mapping bug" vs "mapping verified correct" | 150 Q1/Q6 vs 152 §2.2-Fix4 | Mapping correct; convergence failure |
| 3 | DETACH_PSR_FPN=False "fixes gradient flow in V3" vs env-var never read | 152 §1.4, 153, 155, 156-Q5/Q41 vs code | **No-op** (F-1); V3 runs detached |
| 4 | "Head repair never wired" vs repair present in HEAD's PSRHead | 150 §0.3 vs model.py:1604–1624 | Repair now hardcoded in tree; running-process state unconfirmed (F-10) |
| 5 | Post-activation +4608 vs +384 | 150/151/152/156 vs 150_SOTA_STATUS_V5 | Unresolvable remotely (F-7); avoid quoting a specific value |
| 6 | V3 expected F1 >0.78 vs 0.71–0.74 | 151 §3.5, 155, 156-Q83 vs 140 §-1/150 | **0.71–0.74** (Kendall-only) |
| 7 | LOO ±0.0158 vs ±0.0163 | 150/SOTA_STATUS vs loo_stratified.json | **±0.0163** (F-8) |
| 8 | Per-recording median up 5.82° vs 7.58° | 151 §1.1, 154 §2.4, 156-Q63 vs 140 row 3 / pose memo | **7.58°** (all-16); 5.82 forbidden as headline |
| 9 | "~15° uncited SOTA — beats it" vs "do not cite" | 151 §1.4, 155, 156-Q62 vs 154 §2.4 | Delete ~15° everywhere; first-baseline only |
| 10 | "BEATS WACV 0.95" vs citable 0.838/0.641 | 150_SOTA_STATUS_V5, 155 vs 154 | Use 0.838/0.641; retire ~0.95 |
| 11 | full_eval_ep18_v2 "must be committed" vs committed | 150 §0.2 vs git ls-files | Committed (F-6); action done |
| 12 | Procedure-order cost "not quantified" vs oracle 0.5947/0.8807 | 150 Q17 vs 156 Q45 | Quantified — 32-point oracle gap; cite it |
| 13 | D4+D1R "decisive/verified" vs n_videos=3 + "marginal benefit" verdict | 150/154 headline vs d4_d1r/retune/verdict.json | Keep direction; add n=3 + post-hoc-sweep caveats (F-5) |
| 14 | Fix commits e618d929a etc. vs hashes absent from repo | 152 §6, 155 Repro vs git cat-file | Re-key citations to existing hashes / file+line (F-4) |
| 15 | "9 fixes" vs 10-row fix table | 152 title vs 152 §6 | 9 real fixes + 1 no-op launch-script row |
| 16 | TCN-on-ConvNeXt "pending" vs 0.0723 committed | parts of 156 Q26/Q56 vs temporal-probe JSON | Done and failed (0.0723 < 0.2288 baseline); only TCN-on-MViT pending |
| 17 | Single-task det ETA "epoch 24+" vs "epoch 43+" | 150_SOTA_STATUS_V5 vs 150/151 | Different snapshot times; 🔒 unverifiable; quote with timestamps |
| 18 | "model 29.7% worse than persistence = not learning" vs null-delta learning | 154 §3.1 row 6 vs psr_null_delta_table | Both true under different nulls; name the null in every row (F-11) |
| 19 | Outlier-exclusion numbers "7.39 fwd / 9.14 fwd" swapped | 156 Q65 vs bootstrap/154 | with-outlier 9.14 fwd; excl. 8.46 fwd / 7.39 up |
| 20 | 156 commit messages vs file contents | git history | File overwritten; §2 lost; §1/§4/§9 recovered (F-2) |

---

## §10. Final Synthesis — The Single Most Important Question, Answered

**The question (150 §6 / 152 §7 / 154 §7): "Can we prove multi-task HELPS after all 9 fixes + V3 repair + MViTv2-S fine-tuning?"**

**Direct answer: No — not by Jul 20, and the documents should stop implying otherwise. What can be proven by Jul 20 is stronger for this paper: a per-head, denominator-controlled measurement of what four tasks cost one backbone, with every failure causally attributed and every fix committed in a public repository.** "Multi-task helps" requires multi-task ≥ single-task under identical everything; zero such comparisons exist (the only working head, pose, has an explicit memo withholding the claim), and the 2×2 that decides it is a post-freeze program.

**The decisive-test protocol (150 §8), updated by this audit:**

1. **PSR eval when V3 plateaus** (30 min, cached logits; same threshold convention pre/post): expected 0.71–0.74.
   - 0.71–0.74 → Kendall weighting confirmed as a real-but-small factor; proceed to the attributed head-repair run (repair is already in tree, F-10; add resume-exclusion for `output_heads`, wire F-1 first).
   - ~0.70 flat → publishable negative; head starvation (and possibly transformer variance) is the binding constraint; the 1-hour activation diagnostic designs the next run.
   - >0.78 → **run the workstation attribution check before interpreting** (F-10); do not claim the repair validated without it.
   - <0.65 twice consecutively → abort and restore per 140.
2. **Single-task ConvNeXt detection eval at plateau** (1 day, full-38k, identical protocol/conventions as D3):
   - ≥0.5 → multi-task cost is real and large; the cost claim gets its clean denominator; near-SOTA vs WACV 0.641 possibly achieved same-backbone.
   - 0.3–0.5 → partial capability; cost reportable with architecture-capacity caveat.
   - <0.3 → ConvNeXt-Tiny (in this detection-head configuration) is the limiting factor; **delete the multi-task-cost narrative** rather than caveating it.
3. **Joint interpretation** (150 §8's three branches, corrected): the pair (V3-band, single-task-band) selects among "multi-task costs are measurable and partially recoverable" / "the backbone-head configuration, not task count, binds detection" / "PSR's constraint is the head, not the weights" — all three are publishable conclusions for the measurement paper.

**Realistic Jul 20 submission bundle (all committed or in-flight):** pose 9.14°/7.78° first baseline (+ search report to commit); PSR 0.6788/0.7018 with the full null analysis + V3's Kendall number; detection 0.00009-with-causes + single-task denominator + D1R 0.995 ceiling + D4 ladder (n-caveated); activity 0.0236 → probes → 0.3810 arc; 4 pathology exhibits including the one this audit added (F-1); 9 fixes with public commits; 20-item contradiction ledger resolved (§9). That is the paper. The best-case numbers (PSR 0.78+, activity 0.45+, detection-MT 0.3+) are the post-freeze roadmap — real, funded by evidence, and honestly labeled as future work.

**Highest-priority actions emerging from this audit, in order:**
1. Wire `DETACH_PSR_FPN` for real (one line in config.py + config-dump visibility) before any V4 launch (F-1).
2. Workstation check: does the V3 process have the repaired `PSRHead`? (2 min; decides V3's label and interpretation; F-10.)
3. Confirm the AAIML deadline (5 min, DESK — still documented nowhere; gates the entire schedule).
4. Re-commit the recovered 156 sections (§1/§4/§9 + reconstructed §2) so the question set is whole in-repo (F-2).
5. Apply the ten 155 edits (§7) and the ledger resolutions (§9) to the paper draft.
6. Commit the literature-search report gating the word "first" on pose.

---

*End of file 157 — every question in 150–156 answered; every checkable claim checked; every contradiction resolved or escalated. Compiled from a full line-by-line read of documents 150–156, direct verification against `src/models/model.py`, `src/models/psr_transition.py`, `src/training/losses.py`, `src/training/train.py`, `src/config.py`, `src/data/industreal_dataset.py`, `scripts/train_psr_repair_v3.sh`, `scripts/train_psr_repair_wrapper.py`, the committed evidence JSONs under `src/runs/rf_stages/checkpoints/`, the D1R `results.csv`, and the git history of `156_100_DEEP_QUESTIONS.md`.*
