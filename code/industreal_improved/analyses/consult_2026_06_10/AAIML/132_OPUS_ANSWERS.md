# 132 — Opus Answers: Audit of 131 Overview, Top-10 Question Verdicts, Master Plan Amendments

**Date:** 2026-07-06
**Responds to:** `131_OPUS_OVERVIEW_PROMPT.md` (which aggregates 127, 128, 129, 130 + two SOTA-comparison supplements)
**Scope note:** This audit was performed against the **git repository only**. The raw result artifacts listed in 129 §7-§8 (`SOTA_STATUS.md`, `metrics.json` files, `optimal_thresholds.json`, `t3_full_eval.json`, training logs, `/tmp/*.log`) are **not committed to the repo** — they exist only on the training workstation. Everything that could be verified from committed code and data was verified; everything else is marked UNVERIFIABLE-IN-REPO. See §1 and §7.

---

## 1. Verification Audit — What Checks Out, What Doesn't, What Can't Be Checked

### 1.1 VERIFIED (confirmed in committed code/data)

| Claim (from 129/131) | Where verified | Result |
|---|---|---|
| d1r YOLOv8m mAP50=0.995 / mAP50-95=0.861 | `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` epoch 25 | **CONFIRMED** — 0.99484 / 0.86096. This is the only headline metric auditable from the repo. |
| RGB→BGR fix in D1 eval | `src/evaluation/eval_yolov8m.py` (~168, ~330): `[:, :, ::-1].copy()  # BGR for YOLOv8` | **CONFIRMED** |
| RGB→BGR fix in D4 eval | `src/evaluation/eval_yolov8m_psr.py` (~395): `# RGB -> BGR` | **CONFIRMED** |
| numpy `.clamp` → `.clip` fix | `src/evaluation/eval_yolov8m_psr.py` (~308): `np.clip(gt_np[1:] - gt_np[:-1], a_min=0, a_max=None)`; the torch tensor path correctly keeps `.clamp(min=0)` | **CONFIRMED** |
| D1 class alignment 0-indexed, no shift | `src/evaluation/eval_yolov8m.py` (~340): "Both YOLOv8 and the dataset's gt_classes are 0-indexed … No shift needed." | **CONFIRMED** — so mAP=0.0004 is not an off-by-one; see §2 Q6. |
| MonotonicDecoder variable-shadow fix | `src/models/psr_transition.py:28` (`from src import config as _C`), `:134` (`B, T, n_comp = logits.shape`) | **CONFIRMED** |
| PSR per-component head = Linear→ReLU→Linear, final bias −1.0 | `src/models/psr_transition.py:216-237` | **CONFIRMED** — structurally consistent with the dead-head/saturation hypothesis (debate 2.3). |
| Config flags (129 §4) | `src/config.py` | **CONFIRMED** with one correction, below. |
| Position units "DO NOT USE FOR REPORTING" | `src/config.py:853` ("UNIT UNCERTAIN — DO NOT REPORT mm/cm until verified") and `src/evaluation/evaluate.py:1969` ("position_MAE_mm is unreliable — do not use for reporting") | **CONFIRMED in substance**; the 127/129 citation "evaluate.py:1918-1926" is off — the actual line is ~1969. Update the reference before a reviewer chases it. |

### 1.2 CORRECTION to 129 §4 and 130 P1.1/P2.2

`KENDALL_FIXED_WEIGHTS` is **not a config-file toggle**. It is environment-driven:

```python
# src/config.py:96
KENDALL_FIXED_WEIGHTS = os.environ.get('KENDALL_FIXED_WEIGHTS', '0') == '1'
```

P1.1/P2.2's instruction "toggle KENDALL_FIXED_WEIGHTS=True in src/config.py" should read: **launch with `KENDALL_FIXED_WEIGHTS=1` env** — no code edit, no diff, no re-review of config.py. Same pattern applies to `PSR_SEQ_EVERY_N_BATCHES` (env-overridable, default 4). This is good news for P1.1: the ablation is runnable tonight with zero code risk.

### 1.3 UNVERIFIABLE-IN-REPO (exists only on the workstation)

All of the following numbers rest on files that are **not in git**: PSR 0.7499/0.7810/0.7217, activity 0.023/0.028, T3 0.6223, head pose 8.39°/13.52°/26.20°, D1 0.0004, D4 F1=0/POS=0.999, POS=0.968, the 4-CUDA-crash training log, and every `/tmp/*.log`. I have no reason to doubt them — the analysis chain in 127-130 is internally consistent — but **no third party (including a future you after a disk failure, or an artifact-evaluation reviewer) can audit them**. Action item in §7.

---

## 2. Answers to the Top-10 Questions (131 §4 ordering)

### Q1. PSR-3 — Why does Kendall weighting kill the PSR head?

**Verdict: It doesn't. Kendall is a minor contributor; the primary suspect is the PSR head's own architecture/initialization, and the evidence in your own files says so.**

The numbers are inconsistent with the "Kendall suppression" story: log_var_psr=−0.04 is a precision of ~1.04 — a **4-8% down-weight cannot produce exactly-zero RMS gradients on all 11 sub-heads for 3800+ steps**. A multiplicative weight scales gradients; it does not zero them. Exactly-zero per-component gradients with nonzero aggregate RMS (1.88e-03 at shared layers) is the signature of a **dead forward path inside the heads**: `Linear → ReLU(inplace) → Linear(bias=−1.0)` (verified at `psr_transition.py:216-237`). If the ReLU's inputs are negative (std=0.01 init makes pre-activations tiny; any negative drift kills them) the ReLU gates the entire head's gradient to zero, and the −1.0 output bias parks the sigmoid at 0.27 where the focal-loss gradient is small. TI-3's observation ("GELU fully saturated to zero after linear64") independently corroborates numerical collapse, not loss-weighting starvation.

**Consequence for the plan:** P1.1 as written (fixed weights, 5-10 epochs, expect 0.7499→0.82) is likely to **fail**, because it treats the symptom. Do this instead, in order:
1. **1-hour diagnostic, zero training:** forward one batch, print per-component pre-ReLU activations and transition-head outputs. If ReLU inputs are ≤0 across the batch, the head is dead at initialization/current weights — case closed.
2. **Head repair:** re-init output bias to 0.0, replace `ReLU(inplace=True)` with LeakyReLU/GELU in `transition_heads`, warm-start everything else from epoch_18. This is a ~5-line change to `psr_transition.py`.
3. **Then** run the `KENDALL_FIXED_WEIGHTS=1` arm — as the *ablation* that shows Kendall wasn't the cause, which is itself a publishable negative result feeding Pathology 2's reframing.

Note debate 2.3's own resolution said "Architectural fix needed — linear probe, GELU/ReLU test." File 130 P1.1 quietly replaced that with the Kendall toggle. Restore the debate's resolution.

### Q2. PSR-4 — Is D4 F1=0 genuine failure or sparse-signal metric collapse?

**Verdict: Neither "the decoder is broken" nor "the metric collapsed" — the input signal was absent.** Per the PSR specialist's own concession (debate 2.1), the YOLOv8m detector activated on <1% of frames, so the MonotonicDecoder had nothing to fill forward; the all-static output scores F1=0 on transitions while POS/Edit stay near-perfect because ~95% of frames are static. So D4 tells you: (a) transition F1 is the *only* honest metric of the three, and (b) the YOLOv8m→PSR pipeline as configured is not a functioning system — its thresholds (Q48 hysteresis: hi=0.5, lo=0.3, sustain=3, tuned on ConvNeXt logit statistics) were never re-tuned for YOLOv8m confidence distributions (the Paradigm Comparison Reviewer predicted exactly this in debate 2.2). **Before disclosing D4 as "our decoder fails on a SOTA backbone," re-run D4 once with thresholds swept on YOLOv8m outputs.** If F1 stays ~0 after re-tuning, disclose it hard (P2.3). If it jumps to 0.5-0.7, the disclosure text changes completely — from "our decoder is redundant" to "decoder transfer requires threshold recalibration." That's a half-day experiment that could rescue the paper's worst number.

### Q3. PSR-1 — Why does POS=0.968 not translate to good F1?

**Verdict: POS is structurally inflated by the fill-forward prior and must leave the headline.** Debate 2.1 already found the kill shot: a null model that copies the previous frame's state will score near-perfect POS because assemblies are monotonic and ~95% of frames are static — D4 (POS=0.999 with F1=0) *is* that null model, accidentally run. You don't even need a new experiment to prove the artifact; D4 is the proof. **Run the explicit null-model POS anyway (predict-all-zeros and copy-previous-frame, ~30 minutes on cached logits)** so §5.2.1 can print a three-row table: null POS ≈ 0.99x, D4 POS = 0.999, ours = 0.968. That table simultaneously explains the paradox, kills the "POS beats SOTA" temptation, and demonstrates the honesty the paper is selling. POS moves to a footnote/appendix; per-frame F1 and (after P2.6) transition F1 are the PSR story.

### Q4. ACT-1 — Why 0.028 vs MViTv2-S 0.622?

**Verdict: Architectural ceiling, as diagnosed — but the claim is currently unproven in your own evidence chain, and the proof costs 1 GPU-day.** A per-frame MLP genuinely cannot separate take/put/fit of the same object; that argument is sound. But debate 3.1's Reviewer B is right that class-0 collapse *despite* balanced sampling and bias init points at the backbone features, not just the head. The linear probe (P3.4, 1 day) is the experiment that decides whether P1.4 (TCN+ViT, 2-3 days) and P5.1 (MViTv2-S head, 1 week) can possibly work: **if a linear probe on frozen ConvNeXt features also lands at ~0.03, the backbone encodes no action-discriminative signal, and P1.4/P5.1 are dead on arrival because they sit on the same features.** File 130 sequences the probe *after* the expensive runs (P3.4 in "Weeks 5+", P1.4 in Week 4). Invert that: probe first, then decide. Also report the majority-class baseline (~0.0135 chance, class-0 prior higher) next to 0.028 so the paper states plainly that the current number is statistically indistinguishable from the prior — reviewers will compute it if you don't.

### Q5. D-2 — Should the paper claim "detection mAP=0.995, beats SOTA"?

**Verdict: No. Adopt debate 1.1/8.1's resolution without the hedging.** The 0.995 (verified in-repo: 0.99484 at epoch 25) is a separately trained single-task YOLOv8m — it is a *ceiling measurement*, not the paper's model. The defensible structure is exactly the Supportive Reviewer's framing: (1) WACV 2024's 0.838 is beatable in 1 GPU-day, so it's a soft baseline — report this as a finding about the benchmark; (2) the multi-task ConvNeXt-Tiny reaches 0.358 = ~36% of the ceiling (64% multi-task cost — note 131's "64-68% of ceiling" and "-64% cost" phrasings contradict each other; 0.358/0.995 = 36% of ceiling, a 64% cost; fix which one the text uses); (3) that cost, measured cleanly, is the contribution. One caveat before even the ratio is publishable: the 0.358 comes from a 250-batch class-balanced subsample whose sampler over-represents rare classes (debate 1.2, mAP Fairness Reviewer) — until P1.3 lands a full-set number or a 10-seed variance bound exists, the ratio's numerator is soft. Which is Q6.

### Q6. D-1 — What is the true detection ground truth when full eval gives NaN?

**Verdict: There currently is none, and this is the paper's single biggest reporting hole.** Distinguish the two failures cleanly (127 conflates them in places): **D1 mAP=0.0004** is the *authors' pretrained YOLOv8m* evaluated under your protocol — with the 0-index alignment verified in-repo ("No shift needed") and v2's +1-shift giving 0.0, the audit is complete and 0.0004 is real; it means the official checkpoint and your protocol are incompatible (likely label-space or preprocessing mismatch on their side of the export), and the right paper action is to drop cross-eval of the official weights entirely rather than explain a 0.0004. **D3 NaN** is your own model on the full 38k set — that's the one that blocks the 0.358 claim. P1.3 (in-process eval, `EVAL_MAX_BATCHES=0`) is correctly prioritized; add the 10-seed subsample variance run (debate 1.2 resolution) as the fallback if in-process also fails, so the paper can print 0.358 ± σ instead of an unreviewable point estimate.

### Q7. HP-2 — Up-vector: 7.06° vs 13.5° vs 26.20° — which is real?

**Verdict: Treat 26.20° (full eval) as the number of record until P2.4's per-recording breakdown says otherwise.** The 7.06° is from a pre-fix checkpoint era (AC-1 taints it), the 13.5° is a 300-frame subset (no variance bound), and 26.20° is the only full-set measurement. Report the full-eval median with IQR per debate 4.2's resolution. One correction to file 130's success table: **"Head pose up ≤15° by Week 2" is not a thing P2.4 can deliver** — a per-recording breakdown changes *which* number you report, not the model's error. Either the target is "resolve which number is real" (diagnostic, Week 2, achievable) or it's a modeling target (needs a training intervention that isn't in the plan). As written it's a metric that will be missed and demoralize the tracker.

### Q8. A-1 — What does Kendall buy when one head is dead and it needs manual overrides?

**Verdict: As deployed, Kendall is not doing automatic balancing, and the paper should say so — that's a better paper.** The configuration carries KENDALL_HP_PREC_CAP, KENDALL_HP_FIXED_LAMBDA, per-task caps, and a full fixed-weight bypass — five manual guards on an "automatic" mechanism. Combined with AC-5 (the Kendall spiral was never empirically observed) the honest position is: *uncertainty weighting under extreme label sparsity requires bounding, and unbounded Kendall is unsafe* — a theoretical analysis with preemptive guards (the 21_PATHOLOGY_CORRECTIONS reframing), plus the fixed-weight ablation as the empirical leg. Do not write "Kendall automatically balances our four tasks" anywhere; the config file itself is the counterexample, and it's in the repo for artifact reviewers to read.

### Q9. AC-3 — POS paradox: F1=0.7499 vs D4 F1=0 — which is real?

**Verdict: Both, because they measure different pipelines with different metrics — but the comparison as staged is misleading and one control is missing.** 0.7499 is per-frame component-state macro-F1 of the ConvNeXt head with per-component thresholds tuned on val; D4's 0 is *transition-event* F1 of YOLOv8m→decoder. They share neither input nor metric, so "same eval protocol" (131 §8.2) overstates it. The missing cell in the 2×2 is **our ConvNeXt pipeline scored with the transition-F1 metric** — that's P2.6, and it's the single number that decides the PSR narrative: if ConvNeXt transition-F1 lands 0.5-0.7, the story is "competitive states, weaker transitions, here's why"; if it lands near 0, then 0.7499 is mostly prevalence-prior fitting (the Adversarial Reviewer's reading in debate 10.3: components >50% prevalence score 0.75-1.0, components <20% score 0.35-0.40). Elevate P2.6 to Week 1 — it's 1 day on cached logits and everything downstream in §5.2 depends on which story is true.

### Q10. AC-6 — All numbers will change after the simple-head run

**Verdict: Correct, and the plan needs a freeze protocol, which it currently lacks.** Training is mid-flight (epoch 25) and every P1/P2 intervention (fixed weights, head repair, TCN+ViT) forks a new checkpoint lineage. Without a freeze rule the paper will mix numbers from ≥3 lineages — which is exactly the AC-1 failure mode (epoch-11 vs epoch-18 contamination) repeating at scale. Adopt: (1) pick a **results-freeze date** (given the AAIML timeline, end of Week 4 in 130's sequencing); (2) whatever checkpoint is best-per-head on that date becomes the reporting checkpoint, named and hashed; (3) re-run *every* reported eval once against it (the 129 §7 inventory is the checklist); (4) any number in the .tex must trace to that run ID. P3.6's epoch-11/18 audit becomes a footnote of this protocol rather than a separate task.

---

## 3. Rulings on the Critical Debate Resolutions (128)

| Debate | Resolution in 128 | Ruling |
|---|---|---|
| 1.1 / 8.1 detection SOTA | 64-68% ratio framing, drop BEATS SOTA | **Endorse** (with the %-of-ceiling vs %-cost wording fix from Q5) |
| 1.2 NaN full eval | Fix subprocess or in-process; 10-seed variance regardless | **Endorse**; the 10-seed run is not optional if in-process fails |
| 1.3 cost framing | "Multi-Task Cost Measurement" section title | **Endorse** |
| 2.1 POS paradox | Disclose as artifact | **Endorse + strengthen**: D4 is already the null-model demonstration; add explicit null-POS table (Q3) |
| 2.2 backbone swap | Controlled ablations | **Amend**: re-tune Q48 thresholds on YOLOv8m stats *before* treating D4 F1=0 as final (Q2) |
| 2.3 Kendall suppression | Architectural fix (linear probe, activation test) | **Endorse the debate's version, not 130 P1.1's version** — head repair before/with fixed-weight run (Q1) |
| 3.1 activity ceiling | Linear probe → TCN+ViT → majority baseline | **Endorse, and enforce the order** — probe first (Q4) |
| 3.3 per-frame reframing | Latency argument, deliberate-design tone | **Endorse with a warning**: the latency contribution only survives review if the per-frame F1 is reported against the majority-class baseline and beats it clearly; 0.110 macro-F1 vs a strong prior needs the baseline row |
| 4.2 up-vector | Median + IQR per-recording, full-eval number of record | **Endorse** (Q7) |
| 6.1 CUDA crash | Bisect batch 3/4/5 | **Deprioritize** — it's an infrastructure footnote, not a paper claim; batch=2 works, effective batch is preserved by accumulation. Spend those GPU-days on P2.6/P3.4 instead. |
| 9.6 / PW-6 F1=0 placement | Standalone disclosure paired with results | **Endorse**: §5.2.1 alongside the PSR results table, not buried in a limitations appendix — but only after the Q2 re-tune decides what is being disclosed |
| 10.5 Kendall spiral | Reproduce or retract empirical claim | **Retract-and-reframe now** — do not spend compute trying to reproduce it before the deadline; the theoretical-with-guards version (Q8) is defensible and honest |

---

## 4. Master Plan (130) — Amendments to the Critical Path

Re-sequenced Week 1-2, cheapest-decisive-experiments first. Changes from 130 in **bold**.

```
Week 1:
  Day 1:   PSR head activation diagnostic (Q1 step 1)      [1 hr, RTX 3060]
           Null-model POS baselines (Q3)                    [~1 hr, cached logits]
           **P2.6 transition-F1 on our predictions (Q9)**   [1 day, cached logits]
  Day 2:   **P3.4 activity linear probe (Q4)**              [1 day, RTX 3060]
           **D4 threshold re-tune on YOLOv8m stats (Q2)**   [0.5 day, RTX 3060]
  Day 3-4: P1.3 in-process full eval (EVAL_MAX_BATCHES=0)   [1 day]
           + 10-seed subsample variance if it fails
  Day 5+:  **PSR head repair (bias 0.0, LeakyReLU/GELU)**
           then resume training w/ KENDALL_FIXED_WEIGHTS=1  [2-3 days, 5060 Ti]
  All week: P4.1-P4.4 writing items (unchanged)
Week 2:  P2.4 up-vector per-recording; P2.5 LOO CV thresholds
Week 3+: P1.4 TCN+ViT — **only if the linear probe clears ~0.10**;
         P2.1 distillation as before; **results freeze end of Week 4 (Q10)**
```

Rationale: 130's Week-1 plan spends its GPU budget on the fixed-weight training run whose premise (Kendall is the bottleneck) the evidence contradicts, while the four experiments that *decide narratives* (transition-F1, linear probe, null-POS, D4 re-tune) cost under 3 GPU-days combined and are pushed to Weeks 3-5+. Decisions before investments.

Success-metric table fixes:
- "PSR F1 ≥0.83 by Week 2-3" — keep the target, but condition it on head repair, not fixed weights alone.
- "Head pose up ≤15° by Week 2" — replace with "up-vector number-of-record resolved (median+IQR)" (Q7).
- "Activity ≥0.10 by Week 4-5" — gate on the linear-probe result; if the probe fails, the honest Week-4 deliverable is the backbone-bottleneck finding, not 0.10.
- Add: "All reported numbers traceable to freeze checkpoint: yes/no" (Q10).

---

## 5. The 8 Honest Disclosures (P4.3) — Concrete Enumeration

P4.3 references "8 items" without listing them. Proposed §5.4 list:

1. **D4 backbone swap**: YOLOv8m→decoder transition F1 = 0.000 (with re-tune result from Q2, whichever way it lands), demonstrating [decoder non-transfer | threshold sensitivity].
2. **POS is a structurally inflated metric** under monotonic fill-forward decoding; null model achieves ≈ [null-POS]; we therefore exclude POS from headline claims.
3. **Per-frame action classification top-1 = 0.028 (clip 16-frame majority)**, statistically indistinguishable from the majority-class prior of [X]; the per-frame MLP is a floor baseline, not a competitive method.
4. **Multi-task detection** reaches 36% of the single-task YOLOv8m ceiling (0.358 vs 0.995); full-set evaluation [was repaired via in-process eval | is reported as subsample mean ± σ over 10 seeds].
5. **PSR head gradient starvation**: per-component sub-heads showed zero RMS gradient for extended training spans; reported F1 partially reflects backbone features + prevalence prior (per-component breakdown in Table X).
6. **PSR per-component thresholds were tuned on the validation set**; leave-one-recording-out CV yields [result], bounding the selection bias.
7. **Up-vector MAE is unstable across evaluation subsets** (7.06°/13.52°/26.20°); we report the full-set median with IQR and per-recording breakdown.
8. **Head-position units are unverified** (HoloLens export ambiguity); no position-error claims are made (code marks position metrics "do not use for reporting").

Plus the two integrity items that belong in §4/§6 rather than §5.4: Pathology 2 is presented as theoretical analysis with preemptive guards (not an empirical observation), and all pre-epoch-18 numbers were invalidated by a broken best-checkpoint metric and re-derived (AC-1/AC-5).

If the paper prints these 8 with numbers attached, it is more transparent than any of the four baselines it cites — that, plus the pathology analysis and the cost measurement, *is* the contribution.

---

## 6. Direct Answers on Narrative (SOTA-8, PW items)

- **Lead**: one multi-task model, four heads, honest accounting — "what four tasks really cost on one backbone, and three training pathologies that any practitioner will hit." Not a SOTA paper; a measurement-and-pathology paper with two strong absolute results (pose forward 8.39° pending HP-1's citation search; PSR per-frame 0.7499 pending Q9's transition number).
- **Abstract numbers**: pose forward MAE, PSR per-frame F1 with paradigm qualifier, detection cost ratio, and the disclosure count. Never 0.995 (not our model), never POS, never 0.028 without its baseline.
- **HP-1**: if the "~15°" source cannot be found, delete the comparison and claim "first reported egocentric head-orientation baseline on IndustReal" — weaker but attack-proof. Do the ego-pose literature pass (debate 4.3) before writing §5.3.
- **PW-1 rename**: yes, uniformly, including the .tex "Activity head" → "per-frame action classification head" (grep both `Activity recognition` and `activity recognition`).

---

## 7. Repository Hygiene — Required for Any of This to Be Auditable

Only **one** of the paper's headline numbers (d1r 0.995) is currently verifiable from the repository. Before the freeze:

1. Commit the small evidence artifacts: `SOTA_STATUS.md`, all `metrics.json` (D1 v1-v3, D3, D4, full_eval_ep18_stream), `optimal_thresholds.json` (both sweeps), `activity_clip.json`, `t3_full_eval.json`, `t3_mecanno_eval.json`. These are KB-scale JSON/markdown — there is no size excuse.
2. Move the `/tmp/*.log` eval logs (129 §8 lists six) into `src/runs/.../logs/` and commit; `/tmp` does not survive a reboot, and four of the ten "read these first" audit files in 129 §16 live there.
3. Fix the two stale references found in this audit: `evaluate.py:1918-1926` → `evaluate.py:1969` (127 HP-3, 129 §13.4), and "toggle KENDALL_FIXED_WEIGHTS in src/config.py" → "env var" (130 P1.1/P2.2, 129 §4).
4. Record the freeze-checkpoint SHA256 alongside the committed metrics (Q10).

---

## 8. One-Paragraph Bottom Line

The strategy documents are coherent and the self-criticism is genuinely strong, but the master plan inverts its own evidence in two places: it buys GPU-days for the Kendall fixed-weight hypothesis that the gradient evidence contradicts (the PSR head is architecturally dead — ReLU/bias saturation, verified plausible in `psr_transition.py:216-237` — so repair the head first), and it defers the four cheap decisive experiments (transition-F1 on our predictions, activity linear probe, null-model POS, D4 threshold re-tune — under 3 GPU-days total) that determine which paper can honestly be written. Detection's 0.995 is real and verified in-repo but is a ceiling measurement, not a claim; POS leaves the headline; activity's 0.028 needs its majority-class baseline printed beside it; the up-vector number of record is the full-eval median pending P2.4; and a results-freeze protocol plus committing the (currently workstation-only) evidence artifacts are prerequisites for the honesty section §5.4 to be worth the name. Run the Week-1 sequence in §4, and every downstream writing decision in P4 resolves itself from the results.

---

**End of 132. Next expected files: 133 (Week-1 experiment results) feeding the §5 disclosure numbers.**
