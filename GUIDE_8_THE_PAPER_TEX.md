# GUIDE 8 — THE PAPER: `popw_paper_improved.tex` from `\todo` to submission-ready

*Dedicated companion to the paper. Answers every question in doc 50, maps every placeholder
to its source, and supplies paste-ready prose that contains **no results** — every number
stays a `\popwres`/`\todo` placeholder while training runs. Read after GUIDES 1–7.*

> **Ground rule honored throughout:** I never fill a result. All paste-ready blocks use
> `\popwres`/`\todo` where a number goes. You own the numbers; I own the scaffolding.
>
> **Paper is at three paths** (`code/popw_paper_improved.tex`, the consult-dir copy, and a
> `…copy.tex`). **Pick ONE canonical file** before editing or you'll diverge. This guide
> assumes `code/popw_paper_improved.tex`.

---

## PART 0 — The one-screen answer to doc 50

**Is a benchmarkable paper achievable from here? Yes.** Not by beating YOLOv8m — by proving
the thesis (one shared backbone, 4 benchmarkable heads, one forward pass, efficient, with
FiLM) with honest numbers + the two ablations. Your blockers are no longer code (all 7
guides + 5 patches are implemented per doc 49); they are **(a) running the ablations** (now
possible — Phase B is unblocked) and **(b) honest framing of detection**. Everything else is
writing, which you can do now with placeholders.

**The single most important strategic correction:** stop spending compute on "breaking the
0.207 ceiling." It is a diluted metric (honest = 0.304 pc) on a fine-grained-state task. The
paper does not need a better detector. It needs the **ablations** and the **other heads
alive**. Re-allocate every hour from detection-tuning to ablations + activity/PSR validation.

---

## PART 1 — Minimum publishable paper & per-head "benchmarkable" bars (answers Q1.1–1.3)

**Minimum publishable (all achievable):**
1. 4 heads with non-trivial test numbers (detection ✅0.304pc, head pose ✅9°, activity →rf3/PhaseB, PSR →validate).
2. Efficiency table (params/FLOPs/FPS) — computable now.
3. **Ablation A** (single-task vs multi-task) — run via Phase B.
4. **Ablation B** (FiLM ladder) — run via Phase B.

**Ablations are NOT optional and NOT blocking-on-detection.** They are the scientific core;
run them now (Part 5). The paper is publishable with modest absolute numbers *if* the
ablations + efficiency + honest framing are present.

**"Benchmarkable" threshold per head (your Q1.3 table, answered):**

| Head | Alive (min) | Credible | Strong | Your status |
|------|-------------|----------|--------|-------------|
| Detection (ASD) | mAP50_pc ≥ 0.25 | ≥ 0.35 | ≥ 0.45 | ✅ 0.304 (already benchmarkable) |
| Activity | clip Top-1 ≥ 7% (5× the 1.35% chance) | ≥ 15% | ≥ 30% | ⏳ needs rf3 / Phase B |
| PSR | F1(±3) ≥ 0.20 + >10 unique patterns | ≥ 0.40 | ≥ 0.506 (=STORM-PSR) | ⏳ validate first (Part 4) |
| Head pose | report it | MAE ≤ 20° | ≤ 12° | ✅ 9.13° (strong, uncontested) |
| Body pose | IKEA only, PCK ≥ 0.5 | ≥ 0.7 | ≥ 0.85 | ➖ conditioning-only on IndustReal |

If activity ≥ 7% and PSR ≥ 0.20, you have **4 alive heads** — enough.

---

## PART 2 — Venue (answers Q1.4 / Q7)

- **Realistic & honest target: WACV or BMVC, or a CVPR/ICCV workshop** (egocentric / industrial
  / assistive vision). IndustReal itself is WACV 2024 — WACV is the natural home.
- **The contribution is a systems/efficiency MTL result**, not a SOTA-accuracy result. That fits
  WACV/BMVC well; reviewers there value working unified systems + efficiency + honest ablations.
- **CVPR/ICCV main track** is worth attempting **only if** you land: (a) positive transfer or a
  clean accuracy–efficiency Pareto on ≥2 heads, (b) both ablations significant beyond seed
  noise, and ideally (c) both datasets (IndustReal + IKEA ASM). Without those, main-track risk
  is high; don't burn the submission window on it.
- **Detection gap is not disqualifying** at WACV/BMVC if framed as fine-grained state ID +
  efficiency (Part 3). It would be fatal only if you claimed to beat YOLOv8m.

---

## PART 3 — Honest detection handling (answers Q2.8) — paste-ready

**Decision: report BOTH metrics and frame dilution as a finding (your option A+D).** Never put
0.207 next to 83.80 unqualified. Paste this into the detection results discussion (numbers stay
placeholders):

```latex
\noindent\textbf{Detection metric convention.} The IndustReal ASD benchmark averages AP over
all \num{24} assembly-state channels. In any subset, several states have no validation
instances; the COCO-\num{24} mean (\popwres{}) therefore averages in zero-GT channels and the
background channel, diluting the headline. We additionally report the present-class mean
(\popwres{}), averaged only over the \popwres{} channels with ground-truth instances, which
reflects performance on the states actually evaluable. We emphasize that the YOLOv8m reference
(\num{83.80}\% mAP@0.5) was trained with COCO pretraining and ${\sim}$\num{260000} synthetic
images plus real fine-tuning, whereas \popw{} uses ImageNet pretraining and real data only;
the comparison is therefore one of \emph{data budget}, not architecture. Qualitatively
(Fig.~\ref{fig:det-confusion}), residual error concentrates on assembly states that differ by
a single component, i.e.\ the task is fine-grained state discrimination rather than
localization (localization recall is high; see Table~\ref{tab:industreal-headline}).
```

Use the **24×24 detection confusion matrix** you already implemented (`evaluate.py:1695`) as
`Fig.~\ref{fig:det-confusion}` — it is the figure that turns "low mAP" into "interesting
finding." On the **12 dead classes**: state plainly they are the <50-instance tail (data
scarcity) and their Hamming-neighbor confusion; do not present it as a bug.

---

## PART 4 — Scope decisions (answers §3, §4, §6, §9)

### 4.1 Body pose (Q6) — **conditioning-only on IndustReal**
IndustReal has no body-keypoint GT (your code says so). On IndustReal, present body-pose as the
**PoseFiLM conditioning mechanism** (Contribution 2), not a benchmark row. PCK rows belong to
IKEA ASM *only if* you run it (4.4). **Edit the paper** so the IndustReal tables don't imply a
body-pose number you can't produce.

### 4.2 Activity (Q3) — **train it; you do NOT need to pass the rf2 gate**
Two paths, pick one:
- **(Recommended) Phase B**: cache rf2 backbone features, train the activity head on the cache
  (fast, isolates it, gives the single-task arm for Ablation A for free). Doesn't touch the live
  rf2 run.
- **rf3**: let the gate advance (now honest: `det_mAP50_pc ≥ 0.28`, which 0.304 already passes)
  — `stage_manager` will launch rf3 with `train_act=True`.
Realistic after 15 ep / 35% subset or Phase B: **clip Top-1 ~10–25%**. That clears the "alive"
bar. If it collapses to 1–4 classes, lower imbalance pressure (sampling), keep LDAM off.

### 4.3 PSR (Q4) — **validate before claiming**
The constant `1.546e-08` loss means PSR has **likely never actually trained** (no gradient).
**Before writing any PSR number: run a Phase-B PSR-only overfit** on cached features (50
sequences). If F1 climbs → report it. **If it cannot learn → present PSR as architecture +
"preliminary/future work" and remove the PSR result rows** (don't ship fabricated/zero
numbers). Honesty here protects the whole paper.

### 4.4 IKEA ASM (Q9) — **optional secondary; don't block on it**
Costs 2–3 days and the eval path is untested. **Primary paper = IndustReal (4 heads).** Treat
IKEA ASM as a stretch that adds body-pose PCK + cross-dataset generalization *if* time remains.
Write the paper so it stands on IndustReal alone, with IKEA ASM as additive.

---

## PART 5 — The ablations: run them NOW via Phase B (answers §7)

Phase B is unblocked (patches P1–P5 applied per doc 49), so the ablations no longer wait on a
better detector. They share the **frozen rf2 backbone** → cheap and clean.

**Ablation A (single vs multi-task), minimum viable:** you don't need all 5×single-task runs.
Run **detection and activity** single-task vs multi-task on the identical frozen backbone. That
is the reviewer-sufficient core. (Head pose single-task is a cheap bonus.)

**Ablation B (FiLM ladder):** run in Phase B on the activity head — `No FiLM → PoseFiLM →
HeadPoseFiLM → Both` (flags `use_hand_film`, `use_headpose_film` exist). This is the test of
your headline novelty; it needs activity to run, which Phase B provides.

**Be robust to the outcome (critical):** if multi-task ≥ single-task → claim positive transfer;
if ≈ → claim accuracy–efficiency Pareto; if < → report the interference honestly and lean on
efficiency + FiLM. **All three are publishable** (Standley ICML'20; PCGrad NeurIPS'20). Write
the conclusion (Part 8) so it holds under any of them.

---

## PART 6 — Placeholder → source map (refines doc 50 §10.1)

For each, the exact filler + minimum acceptable + when. "Now" = no training needed.

| Paper element | Fills from | Min acceptable | When |
|---|---|---|---|
| IndustReal ASD mAP (b-boxed / all-frames / @[.5:.95]) | full-test `evaluate.py` (`det_mAP50`, `det_mAP50_pc`+`n_present`, `det_mAP_50_95`) | report honestly | after rf2 eval |
| IndustReal head pose (fwd/up/pos MAE) | rf2 eval (`forward_angular_MAE_deg` ✅; add up/pos logging) | ≤20° fwd | now-ish (rf2) |
| IndustReal activity Top-1/Top-5 | rf3 or Phase-B clip-level eval | Top-1 ≥7% | Phase B / rf3 |
| IndustReal PSR F1(±3/±5)/POS | Phase-B PSR (after validation, Part 4.3) | F1 ≥0.20 | Phase B |
| Efficiency params | `count_parameters` (`model.py:2169`) | exact | **now** |
| Efficiency GFLOPs/FPS (batched+streaming) | `efficiency_report.py` | exact | **now** (no training) |
| Ablation A (heads table) | Phase-B single vs multi (Part 5) | det+act rows | Phase B |
| Ablation B (FiLM table) | Phase-B FiLM ladder | 4 rows | Phase B |
| Ablation backbone (ResNet-50 vs ConvNeXt) | optional; cite if not run | can mark "future" | optional |
| Ablation MTL weighting | `KENDALL_FIXED_WEIGHTS` toggle | optional | optional |
| Contributions claim 3/4 `\todo` | Ablation tables once filled | — | after ablations |
| §3.4 VideoMAE `\todo` | **drop or mark future** (never trained) | — | decision |
| §7 Failure cases `\todo`×8 | **draftable now** (Part 8) | prose | **now** |
| Conclusion (commented out) | **paste Part 8** | prose | **now** |
| Abstract (missing) | **paste Part 8** | prose | **now** |
| Architecture figure | draw from `model.py` forward | — | now |
| Det/act confusion figures | `evaluate.py` PNGs (implemented) | — | after eval |

---

## PART 7 — Code↔paper drift to reconcile (DECISIONS FOR YOU — I did not change these)

The paper's Implementation Details describe a setup that differs from the committed code. Pick
ground truth and make them agree (these are method facts, not results, so they're yours to
finalize against your *actual* final run):

| Paper says | Code says | Action |
|---|---|---|
| Optimizer **Lion** (L464, L478) | `USE_LION=False` → **AdamW** | Decide; make both match. (Paper is internally consistent on Lion; code uses AdamW.) |
| **Batch size 1** ×32 accum (L462, L480) | rf presets **batch 4** ×8 accum | Set to what you actually ran (effective batch 32 either way). |
| Mixup/CutMix **applied** (L489) | `USE_MIXUP=False`, `CUTMIX_ALPHA=0` (label-corruption) | Change to "disabled (logit-mixing corrupts labels)". |
| **3-stage epoch schedule** 1-5/6-15/16-100 (L500-503) | RF1–RF10 stage-manager curriculum | Rewrite to the curriculum you actually used. |
| **LDAM-DRW @ epoch 60** (L502) | `USE_LDAM_DRW=False` (CB-Focal) | Fix to CB-Focal + label smoothing. |
| **PSR T=4, every 10 steps** (L510) | `PSR_SEQUENCE_LENGTH=8`, every 2 | Fix to actual. |
| **w/ VideoMAE** row 75.3M (L684) | never trained, `USE_VIDEOMAE=False` | Drop the row or mark "future work". |

A paper that misdescribes its own method is a reviewer red flag — fix these before submission.

---

## PART 8 — Paste-ready prose (RESULT-FREE; numbers stay placeholders)

### 8.1 Abstract (insert after `\maketitle`)
```latex
\begin{abstract}
Assembly understanding requires interpreting egocentric video across several levels
simultaneously: which assembly state is present, where the worker's head is oriented, what
action is performed, and which procedure step has completed. State-of-the-art systems address
these with separate specialist models, incurring redundant computation and no shared
representation. We present \popw{}, a unified architecture that performs assembly-state
detection, head-pose estimation, activity recognition, and procedure-step recognition in a
single forward pass over a shared ConvNeXt-Tiny\,+\,FPN backbone (\popwres{} parameters),
with a two-stage FiLM mechanism that conditions high-level features on pose and gaze. To train
heterogeneous tasks stably on commodity hardware (a single 12\,GB GPU), \popw{} routes
temporal-head gradients away from the shared trunk and weights tasks with homoscedastic
uncertainty. On IndustReal, \popw{} attains \popwres{} present-class detection mAP, \popwres{}
activity Top-1, \popwres{} procedure-step F1, and \popwres{}$^\circ$ head-pose error, while
using \popwres{} fewer parameters and a single backbone pass versus an ensemble of specialists
(\popwres{}). Ablations isolate the contribution of multi-task sharing and FiLM conditioning.
\popw{} shows that unified egocentric assembly understanding is feasible and efficient without
catastrophic task interference. Code and configurations will be released.
\end{abstract}
```

### 8.2 Conclusion (replace the commented-out block ~L1099-1111)
```latex
\section{Conclusion}
\label{sec:conclusion}
We introduced \popw{}, a unified multi-task architecture for egocentric assembly understanding
that performs assembly-state detection, head-pose estimation, activity recognition, and
procedure-step recognition in a single forward pass over a shared backbone. Rather than
pursuing per-task state of the art, \popw{} targets a practical regime: competitive per-task
accuracy at a fraction of the parameters and compute of separate specialists, trainable on a
single commodity GPU. Two design choices make this possible: two-stage FiLM conditioning for
cross-task information flow, and gradient-isolated temporal heads with homoscedastic
uncertainty weighting, which together avoid the interference that destabilizes naive joint
training. Our ablations (Sec.~\ref{sec:ablation}) quantify the effect of multi-task sharing and
of FiLM conditioning. \popw{} attains \popwres{} on detection (present-class), \popwres{} on
activity, \popwres{} on procedure-step recognition, and \popwres{}$^\circ$ head-pose error,
at \popwres{} of the parameters of a specialist ensemble. Limitations (Sec.~\ref{sec:limits})
include real-data-only detection training and single-GPU constraints. Future work includes
synthetic-data pretraining for detection, a Kinetics-pretrained video stream for activity,
and evaluation across additional assembly datasets.
```

### 8.3 Limitations (add as `\subsection{Limitations}\label{sec:limits}` before Conclusion)
```latex
\subsection{Limitations}
\label{sec:limits}
\popw{} is trained on a single 12\,GB GPU, which constrains batch size and precludes a
Kinetics-pretrained video encoder for activity recognition. Detection is trained on real data
only; the YOLOv8m reference additionally uses ${\sim}$\num{260000} synthetic images, so the
absolute detection gap reflects a difference in data budget rather than architecture.
IndustReal provides no body-keypoint annotations, so body pose is used solely as a conditioning
signal rather than a supervised output (body-pose metrics are reported on IKEA ASM only). To
avoid gradient interference observed in naive joint training, the activity and procedure-step
heads consume stop-gradient backbone features with cross-task conditioning; we therefore report
a stable, conditioning-based multi-task design rather than full joint representation learning.
Finally, several assembly states with very few training instances remain difficult, and we
report results from \popwres{} seed(s).
```

### 8.4 Failure cases (replace the speculative `\todo` bullets in §7 with honest, hedged prose)
Keep your three categories but replace fabricated numbers with `\popwres{}` and soften
unverified mechanisms (e.g. "we hypothesize" / "we observe"). Do not state an occlusion or
rare-class degradation number you have not measured.

---

## PART 9 — Day-by-day execution plan (answers Q14.1 / Q10.2)

Re-allocate compute away from detection-tuning. Recommended order (single GPU):

1. **Now (no training):** run `count_parameters` + `efficiency_report.py` → fill efficiency
   table; draft **abstract, conclusion, limitations, failure cases** (Part 8); reconcile drift
   (Part 7); set the canonical `.tex`.
2. **Let rf2 finish → full-test eval:** fill detection rows (both metrics + n_present) + head
   pose (add up/pos MAE logging) + the **detection confusion figure**.
3. **Phase B (the high-ROI block):** cache rf2 features → (a) train **activity** (single-task
   *and* multi-task = Ablation A det+act), (b) **FiLM ladder** = Ablation B, (c) **validate
   PSR** (overfit; report only if it learns).
4. **Fill ablation tables + per-task table.** Write the honest cross-method comparison.
5. **Optional / if time:** rf3 end-to-end for the "curriculum" narrative; IKEA ASM for body
   pose + cross-dataset; multi-seed (≥ headline tasks).
6. **Submit** with: 4 alive heads, efficiency, both ablations, honest detection framing,
   limitations. That is a complete WACV/BMVC paper.

---

## PART 10 — If time runs short (answers Q12.3 / Q14.2) — the robust cut

Produce a paper **regardless of which unknowns resolve**:
- **Must-have (don't cut):** detection (honest, ✅), head pose (✅), efficiency (computable),
  **Ablation A det+act**, **Ablation B FiLM**, limitations, confusion figure.
- **Nice-to-have (cut first if needed):** PSR numbers (→ "preliminary"), IKEA ASM (→ future),
  rf3 full curriculum, multi-seed (→ "single seed; multi-seed in camera-ready"), VideoMAE row.
- **Never ship:** a fabricated/zero PSR or body-pose number; an unqualified 0.207-vs-83.80.

The robust spine = **shared backbone + 1 forward pass (C1) + efficiency (C2) + head pose
(uncontested) + detection (honest) + the two ablations (C4/C5)**. Activity "alive" strengthens
it; PSR/IKEA are bonuses. This spine does not depend on any of the 7 unknowns in doc 50 §14.2.

---

## PART 11 — Risk mitigations (answers §13)

| Risk | Mitigation |
|---|---|
| R1 detection < target | Already mitigated: honest present-class framing + fine-grained-state finding; detection is one row, not the thesis. |
| R2 activity crashes | Train it in **Phase B** (frozen backbone, batch 128) — far more stable than joint; fall back to a linear head on cached features. |
| R3 PSR can't train | Validate first (Part 4.3); if it can't, scope to architecture + future work and **drop the rows**. |
| R4 GPU/outage | `crash_recovery.pth` + checkpoints exist; archive checkpoints off-box. |
| R6 narrative dominated by det gap | Lead with **efficiency + FiLM + head pose**; detection after, framed (Part 3). |
| R7 single-seed | Acceptable for submission with an explicit note; promise multi-seed for camera-ready. |
| R8 novelty challenge | Foreground the two-stage FiLM + stop-gradient stable-MTL design + the unified one-pass system; ablations substantiate it. |

---

## Bottom line
You have a complete, well-structured paper and a working pipeline. The path to "done" is:
**reconcile the method drift, fill efficiency now, finish rf2 + eval, run Phase-B activity +
both ablations + PSR validation, paste the prose in Part 8, frame detection honestly, submit to
WACV/BMVC.** No unknown in doc 50 blocks that spine. Stop optimizing detection; start filling
tables.
