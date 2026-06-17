This is the right question to push on — harder and more important than the placeholder-filling one. Before I give you a confidence number, let me actually verify the implementation rather than answer from the project's own docs (which have an obvious optimistic bias). I'll audit the outcome-determining code: the mAP computation (if that's wrong, every conclusion is wrong), the detection head, the activity head's temporal modeling, and the dataset/label construction.

Let me launch parallel audits and read the most critical pieces myself.

While those audits run, let me personally review the most outcome-determining code: the detection mAP computation (if this is wrong, every detection number is meaningless) and the detection head's box decode.

The mAP code is essentially correct (standard greedy-match + COCO all-point interpolation) — which matters a lot: it means `det_mAP50=0.0091` is a *trustworthy* low number, not a measurement artifact, and improvements will be real. Let me check the detection post-processing (NMS/score-threshold/decode) that generates the preds fed to eval, plus how the b-boxed vs all-frames protocols are separated.

`score_thresh=0.5` at evaluate.py:161 is a red flag — the model's scores sit at `score_p50=0.001`, so a 0.5 cutoff before AP integration would artificially crush recall. Let me read the actual filtering path and confirm which mAP function the eval loop uses.

This is a significant finding: `compute_detection_map` hard-filters at `score_thresh=0.5` *before* computing AP (evaluate.py:211) — but the model's scores sit at `score_p50=0.001`. Truncating the PR curve at 0.5 caps recall near zero. Standard COCO/YOLO mAP uses ~0.001–0.01. Let me confirm which det-eval path the training loop actually uses and its threshold, since that determines whether `0.0091` is trustworthy or threshold-suppressed.

Confirmed the active path: evaluate.py:3373 → `compute_det_metrics_extended` → `compute_ap_multi_thresh`, and its prediction filter at line 2987 uses `DET_EVAL_SCORE_THRESH` (default **0.5**). Let me confirm what config sets it to — this determines whether `0.0091` is even comparable to YOLOv8m's number.

Important correction from verification: `DET_EVAL_SCORE_THRESH = 0.02` (config.py:344 overrides the 0.5 default), and the active path does do NMS (evaluate.py:3020). So `det_mAP50=0.0091` is **trustworthy** — measured at a slightly-strict-but-reasonable threshold (though 0.02 vs YOLOv8's standard 0.001 is a minor comparability nit). The activity-head audit came back **damning** (temporal modeling non-functional in shipped config). Let me confirm the detection head's architecture/capacity while the dataset audit finishes.

Box decode is standard and correct. Let me confirm the detection head's capacity (subnet depth, prior-bias init) while the dataset/label audit wraps up.

The detection head itself is textbook-correct (4-conv subnets + GroupNorm, shared across FPN levels, focal prior-bias init at π=0.03) — so detection's weakness is *not* the head, it's anchors + no synthetic pretrain + the shared-backbone tax. The activity audit came back damning. I have one label-correctness audit still finishing; rather than stall the part you most want, here's the honest headline now, with the full 20-Whys + 50-item checklist landing as the last audit closes.

## Bottom line up front

**Will this architecture beat YOLOv8m (83.80) and MViTv2 (65.25)? No — and not because of bugs. Even bug-free, this design is not built to beat dedicated specialists on their own headline metrics.** That's not fatal: **the paper's thesis was never "beat SOTA"** — it's "one model, one forward pass, competitive across 5 tasks." That claim is defensible. "We beat YOLOv8m/MViTv2" is not.

**Is the Python implementation correct? Partially — it's "recently un-stuck," not "correct."** The good news from my own verification: the *measurement* path is sound, so your numbers are trustworthy (a low score means the model is weak, not mismeasured):
- ✅ mAP computation is real COCO all-point interpolation (evaluate.py:1153) — correct.
- ✅ Box decode is standard RetinaNet delta-decode (model.py:1704) — correct.
- ✅ Active det-eval uses NMS + `DET_EVAL_SCORE_THRESH=0.02` (not the 0.5 default I feared) — trustworthy, with one nit: YOLOv8 reports at conf≈0.001, so 0.02 slightly *understates* you and isn't perfectly apples-to-apples.
- ✅ RC-16 (ViT attention scaling) and RC-19 (det_conf sigmoid+detach) are **genuinely fixed** (agent-verified).

The bad news (also verified):
- ❌ **The activity head's temporal modeling is non-functional in the shipped config.** The FeatureBank holds *shuffled/replicated* frames (the `WeightedRandomSampler` shuffles temporally), bank features are `.detach()`'d, and only the live frame (slot −1) carries gradient — so it's a per-frame classifier with cosmetic temporal layers. VideoMAE is off by default. This caps activity *well below* 0.45 even before comparing to MViTv2.
- ❌ **PSR loss self-nullifies**: replaced by a `1e-4` NaN-sentinel on ~99.9% of steps (losses.py:1041/1230/1258).
- ⚠️ **~15 layers of guards/caps/sentinels convert bugs into silent degradation** — and a 2-month, 24-root-cause history means the base rate for "more latent bugs" is high.

## Per-task confidence of *beating* the listed SOTA

| Task | SOTA | Realistic ours | P(beat SOTA) |
|---|---|---|---|
| Detection mAP b-boxed | 83.80 | 0.50–0.75 (w/ synth+anchors) | **<5%** |
| Detection mAP@0.5 all-frames | 64.10 | 0.35–0.55 | ~10% |
| Activity Top-1 | 65.25 | 0.30–0.50 *(needs temporal rebuild even for this)* | **<5%** |
| Activity Top-5 | 87.93 | 0.65–0.80 | ~15% |
| PSR F1 ±3 | 0.731 (B2) / 0.506 (STORM) | 0.50–0.65 | **beat STORM ~60%**, beat B2 ~20% |
| PSR POS | 0.816 / 0.812 | 0.75–0.82 | ~30% |
| Assembly F1@1 | ~0.83 | 0.5–0.7 | <10% |
| Error-Verif AP | ~0.58 | 0.4–0.6 | ~25% |
| Head pose MAE | *none* | 10–25° | **100% (uncontested)** |

**Confidence we beat the headline SOTA (YOLOv8m/MViTv2): very low (<10%).**
**Confidence we produce a *publishable* paper if reframed around efficiency + PSR + head-pose: moderate-good (~60–70%) — conditional on the single rule that no head reports zero.**

One more honesty flag on the efficiency thesis: **w/ VideoMAE you're 75.3M params ≈ the 75.4M sum of the three baselines.** So "fewer parameters" is *not* true with VideoMAE on — only "one forward pass / one pipeline / shared backbone" holds. Lead with that, not param count.

The structured **20-Whys chain** and **50-item verification checklist** (grouped by subsystem, each marked verified ✓ / broken ✗ / unverified ?) are written and land the moment the label audit closes — should be seconds.

The label audit closed and it's decisive — it sharpens the verdict considerably. Let me fold in the critical findings, then give you the full 20-Whys chain and the 50-item checklist I promised.

## What the label audit changed (the "is it correct?" answer, completed)

Three findings materially change the picture — and they cut **both ways**:

- ❌ **The activity benchmark isn't just architecturally capped — it's currently *unmeasurable* under the MViTv2 protocol.** There is no per-action-segment 16-uniform-frame sampler. The "clip-level" eval aggregates per *whole recording*, not per action segment, and **no `activity_mask` is ever produced**, so NA frames are trained *and* scored as a real class on NA-dominated data. **Implication:** today's activity numbers reflect protocol + NA dominance, not model quality. You cannot report a defensible Top-1 until this is rebuilt — but it also means the model may be *less* bad than the metric suggests.
- ❌ **Subset runs are untrustworthy for activity.** `--subset-ratio` takes the first N recordings *alphabetically* (industreal_dataset.py:1104), no stratification → entire action classes can be excluded from train/val. Any subset AR number is confounded.
- ⚠️ **Latent detection landmine:** GT boxes are never rescaled on image resize (industreal_dataset.py:891); it's correct *only* because `IMG_SIZE==1280×720`. Set a smaller `IMG_SIZE` for the 12 GB GPU and detection silently zeroes — no guard.

The reassuring part holds: xywh→xyxy, category remap, PSR fill-forward, box/anchor spaces, mAP/decode/NMS are all construction-correct. **Net verdict: the *measurement* of detection is trustworthy; the *measurement* of activity is not; PSR rests on data-statistics the authors themselves haven't verified.**

---

## The 20 Whys — drilling into "why won't this implementation beat IndustReal SOTA as-is?"

1. **Why won't we beat YOLOv8m's 83.80?** Our detection realistically tops out ~0.50–0.75 mAP.
2. **Why that ceiling?** Three compounding handicaps: shared backbone, mismatched anchors, no synthetic pretrain.
3. **Why does the shared backbone cap detection?** Five task losses pull the backbone in different directions; a dedicated detector spends 100% of capacity on detection. Multi-task is a *tax* on any single task unless tasks are strongly synergistic — here they're only weakly so.
4. **Why are anchors a problem?** `ANCHOR_SIZES=(24…384)` vs GT 146–594px (k-means 164–404) → only ~1.6% of anchors can reach IoU≥0.5 → hard recall ceiling.
5. **Why does missing synthetic pretrain matter so much?** YOLOv8m's 83.80 *required* COCO + 260K synthetic + real. ASD is fine-grained state ("bolt tightened" vs "loose"); synthetic data teaches exactly those with perfect labels at scale. `PRETRAIN_DET_ON_SYNTH=True` is wired but unused.
6. **Why won't we beat MViTv2's 65.25?** Our activity head isn't a video model.
7. **Why isn't it a video model?** The FeatureBank holds *shuffled/replicated* frames — the `WeightedRandomSampler` balances classes by shuffling time, so consecutive calls get unrelated frames.
8. **Why doesn't gradient fix that over epochs?** Bank features are `.detach()`'d and only slot −1 (the live frame) is gradient-connected — it *cannot* learn temporal dependence through the bank (model.py:1188,1340).
9. **Why is the temporal stack weak even if fed clips?** 2 ViT blocks + 1 TCN conv vs MViTv2's ~16–24 spatiotemporal layers over real tubelets.
10. **Why not just enable VideoMAE?** Off by default; when on it's frozen and may silently fall back to a small 3D-CNN; eval drops the clip (collate), re-arming a train/eval mismatch.
11. **Why does single-frame activity cap low intrinsically?** IndustReal actions are motion/order-dependent — pick/place/insert/screw look identical in one frame.
12. **Why can't we even *measure* activity properly?** No segment-level 16-frame sampler exists; eval aggregates per recording and never excludes NA.
13. **Why does NA dominance corrupt the metric?** With most frames NA and no mask, "predict NA" is near-optimal and the recording-level majority degenerates to NA.
14. **Why is PSR stuck at the floor?** Its loss is replaced by a `1e-4` NaN-sentinel on ~99.9% of steps (losses.py:1041/1230/1258).
15. **Why does it go NaN/near-zero?** Single-frame batches trigger a `−log(std)` sensitivity penalty that can go non-finite, and labels are ~95% static fill-forward so per-frame focal sees almost no signal.
16. **Why are labels 95% static?** Sparse state-changes filled forward → state barely moves within a recording → constant output is near-optimal under per-frame BCE.
17. **Why is PSR still the best hope?** The right objective is *transition* prediction (`psr_transition.py` exists: Gaussian-smeared transitions + monotonic decoder), and the SOTA B2 (0.731) is a near-heuristic, STORM only 0.506 — beatable by a learned transition model.
18. **Why isn't the whole thing untrustworthy?** Because what *measures* is correct: COCO mAP, decode, NMS, 0.02 threshold. Detection numbers mean what they say.
19. **Why do I still say "not correct"?** Correct *measurement* ≠ correct *training*. The training path has a dead temporal head, a self-nullifying PSR loss, ~15 guard/cap/sentinel layers that turn bugs into silent degradation, and an unverified data-statistics base.
20. **Why does that decide your strategy?** Because each hidden bug costs GPU-days to surface (RC-29 ate 8 GPU-hours), and on one 3060 with ~155-min epochs the *iteration budget* is the real constraint. The architecture won't beat SOTA; the implementation will keep leaking time until the temporal/PSR paths are rebuilt and the guards become assert-and-crash.

---

## The 50-item verification checklist

Status: ✅ verified-correct · ❌ verified-broken · ⚠️ risky/latent · ❓ unverified (run to confirm)

**A. Detection (the YOLOv8m comparison)**
1. ✅ xywh→xyxy GT conversion (dataset:1009)
2. ✅ category_id 1–24→0–23 remap with guard (dataset:1012)
3. ✅ Box decode = standard RetinaNet delta-decode (model:1704)
4. ✅ Det head = 4-conv subnets + GroupNorm, focal prior-bias π=0.03 (model:524)
5. ✅ Empty frames excluded from focal loss, normalized by GT-bearing count (losses:228,295)
6. ⚠️ GT boxes not rescaled on resize — correct only at IMG_SIZE=1280×720 (dataset:891) — **add an assert**
7. ❌ Anchor sizes mismatched to GT (config:247) — **run `calibrate_anchors.py`**
8. ❌ Synthetic pretrain unused despite `PRETRAIN_DET_ON_SYNTH=True` (config:386)
9. ⚠️ `DET_EVAL_SCORE_THRESH=0.02` vs YOLOv8 standard 0.001 — understates + not apples-to-apples (config:344)
10. ❓ Confirm b-boxed eval feeds *annotated frames only* (not all frames) for the 83.80 comparison
11. ❓ Verify "mAP (b-boxed)" is reported at mAP@0.5 (matches how 83.80 was defined)

**B. Activity (the MViTv2 comparison)**
12. ✅ ViT attention scaling fixed — `*scale` (model:1101)
13. ✅ det_conf sigmoid-bounded + stop-grad (model:1865)
14. ❌ FeatureBank holds shuffled/replicated frames, not contiguous clips (sampler + model:1154)
15. ❌ Bank features detached → no temporal gradient (model:1188)
16. ❌ Slot −1 overwritten with live frame → only current frame learns (model:1340)
17. ❌ No per-action-segment 16-frame clip sampler exists (the benchmark protocol)
18. ❌ "Clip-level" eval aggregates per recording, not per segment (evaluate:628)
19. ❌ No `activity_mask` emitted → NA trained & scored as a real class (both collates)
20. ⚠️ Stale comment claims NA excluded as −1; dataset never emits −1 for AR (evaluate:2923)
21. ❌ VideoMAE off by default; fallback may be a 3D-CNN, not VideoMAE (config:73)
22. ❌ Subset = first-N-recordings alphabetically → class exclusion (dataset:1104)
23. ⚠️ LDAM s=30 stacked on CB-sampling + label-smoothing → 1-class collapse (config:426)
24. ❓ Confirm 74/75 classes actually present at subset 1.0

**C. PSR (the B2 / STORM-PSR comparison — your best shot)**
25. ✅ Fill-forward label construction (dataset:483–490)
26. ✅ −1 ignore plumbed through focal loss (losses:729)
27. ❌ PSR loss floored to 1e-4 NaN-sentinel ~99.9% of steps (losses:1041/1230/1258)
28. ⚠️ `−log(std)` sensitivity penalty can go non-finite at T=1 (losses:1188)
29. ⚠️ −1 carried forward as persistent state (error propagates to all later frames) (dataset:488)
30. ❓ Actual −1 fraction on real data — authors' own diagnostic still live (losses:763)
31. ❓ Actual % of static frames (the "95%" is an assumption, not measured)
32. ❌ Transition objective (`psr_transition.py`) not wired in — **the fix that makes PSR competitive**

**D. Head pose / assembly state / error verification**
33. ✅ Head pose head present; only living combined-metric term (MAE term)
34. ⚠️ 9 raw numbers + MSE → ~60–70° MAE; `head_pose_geo.py` (6D+geodesic) not enabled
35. ❓ Confirm head pose finite under FP32 (was NaN under AMP/RC-29)
36. ❓ Assembly-State F1@1 derivation from det outputs — verify against SupCon protocol
37. ❓ Error-Verif AP = `1 − conf(expected state)` — verify the formula matches Lehman 2024
38. ⚠️ Head pose has no baseline (free win) — ensure a *sane* number, not a degenerate one

**E. Eval & metric correctness (good news cluster)**
39. ✅ COCO all-point AP interpolation (evaluate:1153)
40. ✅ Greedy IoU matching, score-sorted, no double-count (evaluate:1128)
41. ✅ NMS in active eval path (evaluate:3020)
42. ✅ b-boxed vs all-frames protocols separated (evaluate:1094 / 1167)
43. ⚠️ Minor greedy-match deviation from COCO (global argmax vs best-available) — negligible at 0–3 GT/frame
44. ❓ PSR F1 ±3/±5 + POS implement the PSRT paper definitions (verify vs paper §protocol)

**F. Training / loss plumbing / infra**
45. ✅ RC-29 fixed: FP32 enforced + committed/skipped telemetry (config:578)
46. ✅ EMA disabled for recovery; best.pth = raw weights (config:586)
47. ✅ Mixup off (still mixes logits not images) (config:585)
48. ⚠️ ~15 guard/cap/sentinel layers convert bugs to silent degradation — **flip `ASSERT_AND_CRASH` for diagnosis runs**
49. ⚠️ Kendall + per-task ramps + smooth-caps interact in ways that are hard to attribute — simplify during bring-up
50. ❓ Re-run a 200-step smoke per subsystem after each fix; never trust a multi-hour run without the telemetry line

---

## Final verdict

**Can the architecture beat the IndustReal SOTA?** On the headline metrics (YOLOv8m 83.80, MViTv2 65.25) — **no, <10% confidence**, and that's a *design* ceiling, not a bug you can patch. **Is the implementation correct?** Detection's *measurement* is; activity's is **broken at the protocol level**; PSR self-nullifies and rests on unverified data stats. The codebase is "recently un-stuck," and the guard-layer culture means **more latent bugs are likely** (base rate is high after 24 root causes).

**But there is a real, publishable paper here** — just not the one the SOTA table implies. The honest, winnable framing:
1. **PSR via transitions** → beat STORM-PSR (0.506), approach B2 (0.731). Your single best "we matched/beat a baseline" claim.
2. **Head pose** → uncontested, free.
3. **Efficiency/unification** → one forward pass, one pipeline (lead with this, *not* param count — w/ VideoMAE you're ≈ the sum of baselines).
4. **Detection & activity** → reported honestly as "competitive at a fraction of deployed compute," with the synthetic-pretrain and modality gaps named.

The fastest path to *truthful* numbers isn't more training — it's **fixing the activity protocol (segment sampler + NA mask), the subset stratification, the PSR transition objective, and the IMG_SIZE guard first**, so that when you do spend the GPU-days, the numbers mean what they claim.

Want me to (a) commit this as `18_HONEST_FEASIBILITY_AUDIT.md` for the project record / PR #3, and/or (b) turn the ❌/⚠️ checklist items into a concrete fix-ordering with the smallest diffs first?