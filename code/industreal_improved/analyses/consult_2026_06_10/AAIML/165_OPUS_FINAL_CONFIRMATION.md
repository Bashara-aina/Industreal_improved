# 165 — Opus Final Confirmation

**Date:** 2026-07-07
**In reply to:** File 164 (confirmation brief), auditing against File 157 (prior answers) + repository HEAD.
**Method:** Every F-1…F-12 claim in File 164 §1/§3 was re-checked against the *actual* committed code and documents (file/line/JSON/git-object), not against the fix-log narratives. Where a fix-log and the tree disagree, the tree wins.

**Verdict legend:** ✅ PASS (verified in tree) · ⚠️ PARTIAL (applied but incomplete or with a caveat) · ❌ FAIL (reported done, not actually done in tree).

---

## §0. Direct Answer — "Is there anything we haven't implemented, or wrongly implemented?"

**Yes. Two real gaps and three minor ones. 7 of 12 are clean.**

| Severity | Finding | What's wrong |
|---|---|---|
| 🔴 **FAIL** | **F-8** | Reported "done / no wrong occurrences." The wrong LOO std **±0.0158 is still live in 6 files**; ground truth is **0.0163**. The "fix" commit `11726b63b` only added a log file and edited none of them. |
| 🔴 **PARTIAL** | **F-1** | DETACH env-read + wrapper are correctly in the tree, but the **actual dead-gradient cause is unfixed**. `PSR_GRADIENT_DEBUG.md`'s "Fix 1" (param-freezing bypass, `train.py:779-803`) and "Fix 2" (Kendall PSR-only guard, `losses.py:1756-1768`) were **never applied**. |
| ⚠️ minor | **F-12** | Docs cite preset `ablation_activity_only`; the real name is `ablation_act_only` (`config.py:1694`). A launch with the documented name KeyErrors. |
| ⚠️ minor | **F-7** | "Resolved to +4608" contradicts Opus's original F-7 advice ("don't quote a number until the log is committed"). Still un-auditable in the public repo. |
| ⚠️ cosmetic | **F-9** | Count reported 37 (164) / 35 (F9 log); actual = 37 in the 8 target files. Markers are present. |
| ✅ clean | **F-2, F-3, F-4, F-5, F-6, F-10, F-11** | Verified correct in the tree. |

---

## §1. F-1 through F-12 Verification

### F-1 — DETACH_PSR_FPN no-op → ⚠️ PARTIAL (fix applied, root cause NOT fixed)

- ✅ Env-read present: `src/config.py:1076` — `DETACH_PSR_FPN = os.environ.get('DETACH_PSR_FPN', 'True') != 'False'`.
- ✅ Wrapper post-preset override present: `scripts/train_psr_repair_wrapper.py:39-43` forces `C.DETACH_PSR_FPN = False` when the env var says so.
- ❌ **The real dead-gradient cause is still in the tree.** Your own `PSR_GRADIENT_DEBUG.md` establishes PSR-head grad RMS = 0 in stages 1-2, and attributes it to two mechanisms *other* than DETACH. Both remain unpatched:
  - **Parameter freezing** — `train.py:779-782` (stage 1) and `799-803` (stage 2): `if 'activity_head' in name or 'psr_head' in name:` → `p.requires_grad = False`, with a reinit bypass for **activity_head only**. `psr_head` is frozen unconditionally. Fix 1 (add `psr_head` to the bypass) was **not** applied.
  - **Kendall staging** — `losses.py:1756-1768`: stage 1 and stage 2 both do `prec_psr = prec_psr * 0; lv_psr = lv_psr * 0`, with no `_is_psr_only_batch` guard. Fix 2 was **not** applied.
- **Consequence for the V3 run:** DETACH=False only alters behavior once PSR unfreezes (stage 3, epoch ≥ 16) — which was already the case *before* the fix. In stages 1-2 the head is frozen and `prec_psr=0` regardless of DETACH. So the marginal effect of the F-1 fix is narrowly "PSR gradient reaches the backbone in stage 3+," **not** "PSR now learns." The V3 F1, whatever it lands at, **cannot be attributed to the DETACH fix** without the two missing patches. File 164 §1 already concedes F-1 is "insufficient"; this confirms it and names the exact two edits that are still owed.

### F-2 — File 156 §2 lost → ✅ PASS
`156_100_DEEP_QUESTIONS.md:463` — `## §2. The Implementation Critique (Q11-20) — RECONSTRUCTED from File-157 F-2 audit`, with Q11-Q20 present. Reconstructed from file 152 as claimed.

### F-3 — Never-predicted class list → ✅ PASS
Live text uses `{1, 13, 16, 19, 23}` in 150/151/152/155. The only surviving occurrences of the old `{1,2,3,14,23}` are in explicit correction-documentation context ("…not {1,2,3,14,23}") in 156, 157, the F3/F5 log, and 164. No live claim carries the wrong list. `147_FINAL_PAPER_NARRATIVE_V4.md:38` was corrected.

### F-4 — Missing commit hashes → ✅ PASS (acknowledged; code present)
Documented in 161/158. Fix code is at HEAD (PSRHead LeakyReLU `model.py:1597-1624`, GT sampler `industreal_dataset.py`, `DET_GAMMA_NEG=2.0` `config.py:832`, `FREEZE_BACKBONE` etc.). Action still owed before submission: re-key 155's reproducibility section to public-repo hashes or file+line (per 157 §7).

### F-5 — 0.6364 on 3 videos → ✅ PASS
`(3-video subset)` caveat present across 150 (7×), 151, 154 (6×), 155, 150_SOTA_STATUS_V5, 144. Confirmed by `F3_F5_CORRECTIONS_LOG.md` and grep.

### F-6 — Evidence dirs committed → ✅ PASS
`git ls-files` returns all seven: `full_eval_ep18_v2/metrics.json`, `d4_retuned/`, `up_vector_v3/…`, `d1_yolov8m_v3/metrics.json`, `psr_optimal_thr_38k/optimal_thresholds.json`, `d3_full_38k/detection_mAP.json`, `activity_mvit_probe/results.json`.

### F-7 — +384 vs +4608 → ⚠️ PARTIAL (resolved, but against advice)
All live `+384` references were replaced with `+4608` (146, 147, 149, 150_SOTA_STATUS_V5, checkpoints/SOTA_STATUS.md; remaining `+384` strings are only in commit-log/resolution docs). **Caveat:** Opus's original F-7 guidance (157 §0 F-7) was to *avoid quoting a specific number* until the log is committed. Hard-selecting +4608 leaves an un-auditable figure in the paper pipeline. Recommendation: commit the `/tmp/train_psr_repair_v3.log` excerpt, or write "moved from a saturated negative regime to large positive activations" with no number.

### F-8 — LOO std 0.0158 → 0.0163 → ❌ FAIL (not implemented)
- Ground truth: `src/runs/rf_stages/checkpoints/psr_loo_cv_stratified/loo_stratified.json` → `loo_improvement_std: 0.0163`.
- The wrong value **±0.0158 is still live** in: `150_SOTA_STATUS_V5.md:33`, `150_MASTER_SYNTHESIS.md:117`, `151_PER_HEAD_DEEP_ANALYSIS.md:87`, `155_FINAL_PAPER_NARRATIVE.md:89`, `147_FINAL_PAPER_NARRATIVE_V4.md:19`, `156_100_DEEP_QUESTIONS.md:363`.
- `F8_LOO_CORRECTION.md` concluded "zero corrections applied — value not present," and its scope-check wrongly asserts "Files 150-156 do not exist in AAIML directory." The commit `11726b63b` ("fix: F-8 …") **changed exactly one file — the log itself** (`+31/‑0`). **This correction has not been made.** File 164's "No occurrences of wrong value found" is incorrect.

### F-9 — Workstation-only markers → ✅ PASS (count wobble)
`UNVERIFIABLE-REMOTELY` appears **37×** across the 8 target files (150:15, 151:4, 152:3, 153:2, 154:1, 155:5, 156:4, 150_SOTA_STATUS_V5:3). The F9 log says 35, File 164 says 37; 37 is correct. Substantive fix done.

### F-10 — V3 loaded repaired code → ✅ PASS (nature-limited)
`F10_V3_PROCESS_STATE.md`: V3 (PID 1901736) started 16:50:36, 18 s after commit `ea6ac30c` (16:50:18); wrapper log shows `DETACH_PSR_FPN=False` and LeakyReLU-active negative post-gelu minima. Sound. Remains 🔒 UNVERIFIABLE-REMOTELY by nature (workstation log). **But** note the F-1 finding: "loaded the fix" ≠ "gradient flows to the head" — the freezing/Kendall path is what actually gates PSR learning.

### F-11 — Null baselines conflated → ✅ PASS
Live 150 text names both distinctly: "persistence null (copy-prev), F1=0.9997" vs "prevalence null (always-positive), F1_null=2p/(1+p)." Disambiguation applied across 150/151/152/154/155/156.

### F-12 — Ablation presets exist → ⚠️ PARTIAL (naming bug)
Presets exist: `ablation_single_task` (1621), `ablation_det_only` (1663), **`ablation_act_only` (1694)**, `ablation_psr_only` (1727), `ablation_pose_only` (1760). Scripts exist: `train_singletask_convnext_det.sh`, `train_singletask_activity.sh`, `run_ablation_suite.sh`. **Bug:** File 157 F-12 and File 164 call the activity preset `ablation_activity_only`; the actual key is **`ablation_act_only`**. `run_ablation_suite.sh` launches det/psr/pose but **not** activity. Anyone following the docs (`--preset ablation_activity_only`) hits a KeyError. Fix the name in the docs and add the activity arm to the suite (or standardize the script path).

---

## §2. 12-Week Plan Assessment → CONDITIONAL (achievable, but the schedule hides a GPU-serialization risk and a broken-attribution risk)

**Achievable IF** two conditions hold; **not** as currently written.

1. **GPU throughput is the binding constraint, and the plan double-books it.** You have 2 GPUs (5060 Ti, 3060). The plan serially requires: V3 PSR + single-task det (running) → 4 single-task baselines (8-12 d) → MViTv2-S fine-tune (14 d) → multi-task V4 (14 d) → *remaining* single-task baselines again (W9-10) → write. At the stated "2 epochs/day" for the large runs and only 2 GPUs, W3-W10 is ~7-8 weeks of wall-clock training that must fit in ~8 weeks with zero slack, zero failed runs, and no re-runs. One crash or one hyperparameter redo blows the buffer. **Add a hard GPU calendar** (which run on which card, which weeks) before committing — right now several items are simultaneously "blocked on GPU."

2. **The PSR arm of the plan is built on the unfixed F-1 root cause.** W7-8 "Multi-task V4 (all 9 fixes + DETACH fix)" is expected to validate "multi-task recovers PSR." But with the freezing/Kendall patches missing (see F-1), V4's PSR head still won't receive gradient in stages 1-2, and the multi-task-vs-single-task PSR comparison will not be clean. **Land Fix 1 + Fix 2 before launching V4**, or the headline 2×2 PSR row is uninterpretable and you burn a 2-week training slot on it.

**Timeline honesty check:** the plan's own §1 marks F-3/F-5/F-7/F-8 "Addressed in SOTA_STATUS update," but F-8 was not (§1 above). The bookkeeping is optimistic; treat "done" flags as "claimed" until diffed.

**Bottom line:** Oct 10 is reachable, but only with (a) an explicit GPU schedule and (b) the two PSR patches landed in the next few days so V3/V4 produce attributable numbers. Writing (W11-12) can and should start *now* for the pathology/diagnosis sections, which need no pending numbers — that is your real schedule buffer.

---

## §3. Final SOTA Verdict — per head

| Head | Honest verdict | Why |
|---|---|---|
| **Detection** | **BEATS — with a hard asterisk** | D1R single-task YOLOv8m 0.995 mAP50 > WACV 0.838/0.641 *cross-architecture* and *single-task*. It is **not** the multi-task system the paper is about (that head is 0.00009, fixes unvalidated). Claim "a fine-tuned detector reaches 0.995 mAP50" — do **not** imply the multi-task pipeline beats detection SOTA. |
| **Head Pose** | **FIRST BASELINE — not "beats"** | 9.14°/7.78° is a credible first number, but the "~15° SOTA" is **uncited**. Per 157 §3/§7, delete the uncited comparison. You cannot claim to beat a number you can't cite. |
| **PSR** | **NEAR — different paradigm, not comparable head-to-head** | 0.6788 (primary) / 0.7018 (per-comp) vs STORM 0.883, which uses procedural transition features, not raw video → paradigm table only, never same row. The "0.78+ with V3" target is **not yet real and now doubly uncertain** given the unfixed gradient path. Report the prevalence null-deltas (+0.097/+0.093/+0.053) as the honest evidence of learning. |
| **Activity** | **NEAR at best, as a single-task video baseline** | 0.0236 multi-task is class-collapse; frozen MViTv2-S probe 0.3810; fine-tune *hoped* 0.45-0.55 vs SOTA 0.622. That is "approaching," never "beating," and only as a single-task backbone baseline — not a multi-task win. |

---

## §4. Criticisms We Missed (fix these before the numbers land)

1. **The PSR gradient is still dead and nobody re-ran the liveness probe after the DETACH fix.** This is the single highest-leverage item. Land `train.py` Fix 1 + `losses.py` Fix 2, then confirm `psr=NonZeroGradNorm` on a stage-1/2 seq batch (train.py head-liveness probe) **before** trusting any PSR number. Until then, V3's F1 is an artifact of stage-3-only training + Kendall-fixed weights, not of a repaired multi-task gradient.

2. **F-8 is unfixed in 6 files — reviewers will diff the LOO std against `loo_stratified.json` (0.0163) and find 0.0158.** One sweep-and-replace fixes it. The bigger lesson: three separate fix-log agents (F-8 especially, and F-1's DETACH claim) reported "done" for work the tree doesn't contain. **Trust `git show --stat`, not the log narrative.** Before submission, diff every "DONE" flag in 158/161.

3. **F-12 preset name (`ablation_act_only` vs `ablation_activity_only`) will silently break the activity ablation launch** and the suite omits activity entirely. This is the one head where you most need the single-task denominator (to make the 0.3810/0.622 story legitimate). Fix the name and the suite now, not on GPU-launch day.

4. **Detection's "beats SOTA" is a cross-architecture, single-task claim** dressed as a system result. A hostile reviewer separates "we fine-tuned YOLOv8m well" from "our multi-task method is good." Pre-empt it: state the architecture and task-set of every SOTA row in the cell.

5. **Every headline PSR activation number (+4608) and the V3 process state are workstation-only.** Commit a timestamped log excerpt into the repo, or the reproducibility section rests on files a reviewer can't open. Same for the 738 MB `best.pth` SHA.

---

## §5. The Final Answer

**No — the paper cannot yet claim to "beat or near SOTA across all four heads," and it should not try to.** What it *can* claim honestly, today, is: (1) a fine-tuned single-task detector at 0.995 mAP50 that exceeds detection SOTA cross-architecture; (2) credible first baselines for head-pose and for PSR-from-raw-video, reported against explicit paradigm caveats and prevalence null-deltas rather than against un-comparable or uncited SOTA; (3) a diagnosis-and-recovery narrative in which "implementation, not multi-task theory, was the killer" is the genuine contribution — and the DETACH env no-op is its sharpest exhibit *because your own advocated method caught it*. That narrative is strong and defensible. But two of the twelve corrections are not actually in the tree: **F-8 is unfixed in six files, and F-1's real cause (PSR head freezing + Kendall zeroing) is still live** — so the multi-task PSR recovery story, which the "near SOTA" claim leans on, is not yet supported by a clean gradient path. Land the two `train.py`/`losses.py` patches, re-run the liveness probe, sweep the 0.0158→0.0163 fix, and rename the activity preset — then the W7-8 V4 run can produce numbers you're allowed to believe. Until then, 7/12 corrections are solid, the plan is achievable only with an explicit GPU calendar, and the honest headline is "diagnosed and recovered," not "beat SOTA."

---

**End of File 165.**
