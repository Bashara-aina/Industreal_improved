# 109 — Round 6: PSR Eval Bug FIXED (F22/F22b) + Answers to the 20 Questions of File 107

**Date:** 2026-07-03
**Context:** After epoch-5 val — RF4 gate passed (combined=0.241, det_mAP50_pc=0.339, act 48/69, pose 8.92°).
**Branch:** `claude/rf4-architecture-consultation-5mnnu5` (restarted from current main; F1–F21 already merged).

---

## 0. Headline: the PSR transition metrics were blinded by TWO stacked bugs — both fixed and verified

**Q1/Q18 answered definitively.** It was neither a one-line reshape nor a protocol
issue — it was two independent bugs in series:

**F22 — grouping misalignment (the crash).** The eval loop collects
`psr_preds_logits` as **per-BATCH arrays [B,11]** but `psr_rec_ids` as
**per-FRAME ids**. The old grouping enumerated batches against frame-ids, filing
whole batch blocks under one frame's recording, so `np.stack` built 3-D
[K,B,11] "sequences" and the metric chain died with exactly your logged error
(`only 0-dimensional arrays can be converted to Python scalars`) → safe-default
zeros. I reproduced the exact error from synthetic data with the old code path.
Additionally, even a correctly-aligned grouping would have been wrong: frames
arrive in **sampler order, not temporal order**, and transition F1 is
meaningless unsorted. The new `_group_psr_by_recording()` flattens per-frame,
aligns ids positionally, collects `metadata.frame_num`, and stable-sorts each
recording temporally (duplicates from the weighted sampler stay adjacent and
create no spurious transitions).

**F22b — MonotonicDecoder dim collapse (the silent one).** The decoder's
blanket `.squeeze()` (added 2026-06-29 for [B,T,C,1] inputs) collapsed a
single-recording batch [1,T,C] → [T,C] → re-expanded to **[T,1,C]: T
independent length-1 sequences.** The monotone fill-forward constraint never
applied across time, and the 3-D output crashed the downstream metrics. Fixed
with explicit dim handling ([B,T,C,1]→squeeze(-1); [T,C]→[1,T,C]).

**Verification (CPU, synthetic 2-recording assembly):** near-perfect predictor
→ F1=1.0/POS=1.0/Edit=1.0; random predictor → F1=0.136. Old code path →
reproduces your production crash verbatim. Two regression tests added
(suite now 24, all passing).

**Two bonus results from the verification:**
- **Random-baseline for the paper:** a random transition predictor scores
  F1@±3 ≈ **0.14** at this transition density — quote it as the null baseline
  next to B2 (0.731) and STORM-PSR (0.901).
- **`psr_pos` is a weak metric** — the random predictor scores 0.95 on it
  (sign-matching mostly-zero diffs is trivially right). Don't gate or report
  on POS without saying this.

**Consequence for file 105's analysis:** the "5/2048 unique patterns" stat and
the "all-ones first frame" observation were measured on RAW sigmoid
binarization, while the decoder — which is what the benchmark metric uses —
was producing garbage. After F22b the decoded states start at all-zero and
fill forward monotonically; **re-measure pattern diversity and transition F1
at the next validation before drawing any architecture conclusions (Q3).**
The next val is the first time PSR's real capability is visible. Doc 105's
"projected 0.05–0.15" may be pessimistic or optimistic — nobody has ever seen
the true number.

Caveat to carry into the paper: the val sampler subsamples frames, so
transition F1 is computed on gapped subsequences (pred and GT on the SAME
subsample — internally consistent, but ±3 tolerance is in subsample units).
For final paper numbers, run one full sequential-order eval pass per
recording (the eval already supports max_batches=0).

---

## Section 1 — PSR (Q1–Q4)

**Q1:** Answered above (F22/F22b). Neither a reshape nor a protocol issue — a
grouping-alignment bug plus a decoder dim bug, both now fixed and
regression-pinned.

**Q2 — detach flip timing:** Don't move it before the fixed metrics produce
two readings (epochs 8 and 11). Decision rule: if PSR F1@±3 at epoch ~11–14
plateaus **< 0.45** while binary acc keeps rising, flip `detach_psr_fpn=False`
at the RF6 boundary AND simultaneously drop `PSR_WEIGHT` 10→2–3 (an amplified
loss suddenly touching the backbone is the actual corruption risk; the F1
snapshot-restore guard protects the accumulation mechanics, not the loss
scale). If F1 is ≥0.45 and climbing, leave the isolation — it's a cleaner
paper story ("PSR learns from frozen shared features") and zero risk.

**Q3 — 5/2048 patterns:** premise is now stale — that statistic came from raw
sigmoid thresholding while the decoder was broken. Re-measure post-F22b. If
decoded diversity is still <15 patterns at epoch 11: the answer is not longer
sequences but the decoder threshold (0.3 — sweep 0.2/0.3/0.4 offline on saved
logits, zero training cost) and per-component transition sensitivity (Q4).
Fill-forward labels do NOT make "never transition" optimal *for the transition
objective* — that's exactly what USE_PSR_TRANSITION exists to prevent.

**Q4 — per-component alpha:** the mechanism already exists
(`per_component_alpha` in `binary_focal_loss`; `_psr_per_component_alpha` in
the criterion) — it needs populating, not building. But note: under the
transition objective the relevant imbalance is **transitions per component**
(roughly equal — each component transitions ~once per recording), not state
prevalence (0.19–1.0). So per-component alpha is likely a small win at best.
Do it at RF6 only if the now-visible per-component transition F1 shows
specific dead components.

## Section 2 — Detection (Q5–Q8)

**Q5 — scores 0.036→0.333:** can't cleanly attribute F1 vs F8 without the
ablation, and it doesn't matter operationally — both push the same direction.
Mechanically: OHEM keeps the untouched anchor sea at bias-init, so a MEAN of
0.333 implies the selected population (positives + hard negatives) now
dominates — exactly the desired signature. Expect 0.45–0.60 by epoch 15–20.
No ceiling from asymmetric gamma: `gamma_pos=0` means positives keep full CE
gradient at every confidence.

**Q6 — mAP50_pc vs mAP50:** report **mAP50_pc as primary** with an explicit
definition ("mean AP over classes with ≥1 GT instance in the eval set, N=15"),
COCO-24 mAP50 as secondary. That's the honest and standard resolution — the
IndustReal WACV baselines evaluate on annotated states; averaging over
never-present channels is a protocol artifact, not rigor. State N per split in
the table caption; done.

**Q7 — 15/24 "detected" classes: misread.** `det_n_present` counts classes
with **ground truth present in the sampled eval set**, not classes the model
finds. Your 250-batch val sample simply never contains the other 9 classes.
The model is not "specializing on the 15 easiest" — the eval never asks it
about the rest. For the paper-final eval, run the full validation set
(EVAL_MAX_BATCHES=0) so all classes with any GT appear; only then read
per-class AP for real gaps.

**Q8 — switching to symmetric gamma at RF8: no.** Symmetric gamma=2 would
re-suppress positive gradient precisely when score refinement matters. If
late-training overconfident false positives appear (watch per-class precision
at epoch ~20), the correct knob is `DET_GAMMA_NEG` 1.5→2.0 — sharpen negative
mining, keep positives at full strength.

## Section 3 — Activity (Q9–Q12)

**Q9 — the 21 missing classes:** almost certainly the tail (verify in one
minute: cross the missing-class list against `_count_act_frames_lightweight()`
counts). Do NOT add class-weighted CE — the balanced sampler already equalizes
exposure; stacking weights re-creates the double-balancing pathology. Accept
the structural macro-F1 cap, and report micro-accuracy + top-5 alongside so
the cap is visible rather than hidden.

**Q10 — temporal path timing:** unchanged from doc 96 §4/doc 102: it's the
Tier-2 upgrade, triggered only if macro-F1 plateaus <0.15 across three
consecutive evals after peak LR (epochs ~12–18). When you do it: train the
temporal stack on the PSR sequence batches (true consecutive frames), keep the
simple head's results as the ablation row, and expect a transient dip — new
parameters at post-peak LR. Don't touch it while macro-F1 is climbing.

**Q11 — loss oscillation 0.33–1.94:** batch-composition variance from the
balanced sampler (tail-heavy batches score high CE). Judge the epoch mean, not
per-500-step samples. Monotonic decline isn't expected until after peak LR.
Nothing residual from F18 — that fix only changes warmup epochs (<3).

**Q12 — top5=0.381 vs macro-F1=0.097:** normal for a recovering long-tail
classifier — features already cluster related actions (right neighborhood,
wrong argmax). Temperature calibration cannot change argmax, so it's not a
calibration fix; part of the gap is genuine verb-group ambiguity. The ratio
closes as features sharpen; report both numbers, and treat top-5 as the
leading indicator (it moves first).

## Section 4 — Kendall & Gates (Q13–Q16)

**Q13 — equilibrium health:** healthy. All lv values are within noise of
`ln(current loss)` given their slow adaptation (doc 102 Q13). lv_act crossing
0 is not an intervention trigger — it just means activity loss dipped below
1.0. The single rule stands: intervene only if |lv − ln(L)| > 1.5 for 5+
epochs.

**Q14 — combined_v2 as gate? Not mid-run.** Also fix the arithmetic: v2's pose
term is `0.15·1/(1+8.92/10) = 0.079`, not 0.015 (the /10 normalizer was
dropped in your calc). Policy: v1 keeps driving best.pth for checkpoint-
history continuity THIS run; the actual go/no-go instrument is the per-head
table (doc 102 §Q14); promote v2 to the selection metric only at a hard
restart boundary (RF6), accepting a best_metric reset.

**Q15 — HP_PREC_CAP: keep it.** Removing it would *activate* the fossil:
effective lv_pose would drop from lv_det (0.125) to the raw −1.000 → pose
precision jumps to 2.7× at peak LR — the exact scenario the cap prevents. The
cap currently costs nothing and insures against pose-loss dips. (Alternative
if you want the fossil gone: `criterion.log_var_pose.data.fill_(0.0)` at the
next restart — but then you still want the cap for dips.)

**Q16 — improvement rate:** the pose term is saturated (~0.136), so all
combined-v1 growth must come from det (0.30·ΔmAP), act (0.35·Δf1) and — newly
visible after F22 — psr (0.20·ΔF1). Realistic: +0.02–0.04 per val through
epoch ~15, THEN a one-time jump at the first post-F22 val if PSR F1 lands
anywhere above 0.05 (e.g., F1=0.10 adds +0.02 by itself). Diminishing returns
start after peak LR, not before.

## Section 5 — Infrastructure & Strategy (Q17–Q20)

**Q17 — start ablations NOW, on the 3060.** Matched-epoch comparison (doc 102
Q20) is the protocol — waiting for "stable" numbers buys nothing because you
compare single-task@epoch-N to multi-task@epoch-N regardless. The 3060 never
crashes and single-task runs fit 12 GB. Every day of delay is a day of ablation
wall-time you can't parallelize later.

**Q18:** Fixed this round (F22/F22b) — before your epoch-8 validation, merge
this branch so that val produces the first real PSR transition numbers.

**Q19 — Xorg: still do it.** "3h without crash" and "F18 fixed the hang" are
both unproven attributions — F18 changes warmup weighting (epochs <3) and the
current run resumed at epoch 5, so F18 is largely inert in it. The historical
crash point (epoch 5, batch ~100) is *ahead* of any run that resumed at 5 only
by ~100 batches — you've already passed it, which weakly favors the
env-mitigation set, but the Xid check (doc 102) on any future crash plus the
xorg.conf pin remain the cheap permanent insurance. Do the pin at the next
natural pause; don't restart a healthy run for it.

**Q20 — venue:** stay the course on the two-paper strategy. The AAIML
submission remains the pathologies paper (F1/F13/F18/F22 are now FOUR
documented case studies of interface-mismatch failures, with before/after
curves — that paper got stronger this week and is robust to metric outcomes).
The benchmark paper (head pose 8.92°/16.6mm as first IndustReal baseline +
single-pass multi-task efficiency + honest per-head numbers) targets a CV
venue/workshop (WACV or ICRA workshop) once RF10 numbers land. Don't collapse
the two into one AAIML main-track submission — they have different reviewers
and different failure modes.

---

## Changes on the branch this round

| ID | File | Change |
|---|---|---|
| F22 | evaluate.py | `_group_psr_by_recording()` — per-frame flattening, id alignment, temporal sort (+ frame_num collection in the eval loop); replaces the crashing inline grouping; dead stale-variable line removed from `_event_f1` |
| F22b | psr_transition.py | MonotonicDecoder explicit dim handling — [1,T,C] no longer collapses into T length-1 sequences |
| — | tests | 2 new regression tests (suite = 24, all pass): end-to-end grouping+decode on synthetic assembly (near-perfect predictor must score >0.8; old path reproduces the production crash), decoder shape contract |

**Action: merge this branch before the epoch-8 validation** — that val then
produces the first real PSR transition F1 in project history, which feeds the
Q2 (detach flip) and Q20 (paper scope) decisions.
