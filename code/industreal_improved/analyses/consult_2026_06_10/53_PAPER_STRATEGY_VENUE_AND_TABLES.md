# 53 — Paper Strategy, Venue, and Table-by-Table Fill Plan (Grounded)

> **Generated:** 2026-06-22 (Opus). Companion to `51`/`52`. Answers docs 48 (Q5–Q7) and 50 (§1, §10, §13–14).
> **Thesis to defend:** *one shared backbone performs several egocentric assembly-understanding tasks in one forward pass, on commodity hardware, with quantified task interference and FiLM conditioning* — **not** "we beat SOTA."

---

## 1. The honest paper, in one paragraph

POPW is a **systems + multi-task-interference study**, not a detector paper. The contribution is empirical: *what does it cost, and what do you lose, when five egocentric assembly tasks share one ConvNeXt-T + FPN backbone instead of five specialists?* The headline assets are (a) **efficiency** — one backbone, one forward pass, measurable today; (b) **no catastrophic interference** — Ablation A, single-task vs multi-task on the identical backbone; (c) **FiLM cross-task conditioning** — Ablation B; (d) **a working head-pose head** (9.13° MAE, no prior supervised baseline); and (e) **detection reframed honestly** as fine-grained single-bit assembly-**state** discrimination (`52` F2), where weak absolute mAP is *expected and explained*, not hidden.

This framing **survives weak detection** because detection becomes the *illustration* of the interference/difficulty story rather than the headline.

---

## 2. Venue call (answering 48 Q7 / 50 Q1.4)

| Venue | Realistic? | Condition |
|---|---|---|
| **Workshop (CVPR/ICCV/ECCV EgoVis, Assembly/Industrial)** | **Yes — primary target** | Current assets + efficiency + one clean ablation |
| **BMVC** | **Yes — stretch-primary** | Both ablations land + activity ≥ ~20% Top-1 + multi-seed-lite |
| **WACV** | Possible | Both ablations clean **and** a second dataset (IKEA) or full-data detection |
| **CVPR/ICCV main** | **No** | Drop this aspiration — it is distorting priorities. 0.30 detection + 2 working heads on one 3060 will not clear a main-conference bar, and chasing it burns the budget. |

**Recommendation:** aim the writing at **BMVC-or-workshop simultaneously** (same paper, the workshop is the floor). Decide venue *after* the ablations resolve, not before.

---

## 3. The reframe that defuses the detection gap (50 Q2.8, R6)

Do **not** put a naked `0.207` next to YOLOv8m's `83.80`. Instead:

1. **Report both** `det_mAP50` (diluted, standard protocol) **and** `det_mAP50_pc` (present-class). State the dilution explicitly: 8/24 channels are zero-GT/background.
2. **Lead the detection subsection with the task definition** (`52` F2): 24 classes are 11-bit assembly states differing by single components; this is fine-grained state recognition, not object detection.
3. **Show the 24×24 confusion matrix** and quantify how much error mass falls on **one-bit-adjacent** states. This converts "low mAP" into "the model captures coarse state, confuses single-component transitions" — a *characterization*, not an apology.
4. **Attribute the gap to budget**, explicitly: YOLOv8m = COCO + 260K synthetic + real, full data; POPW = ImageNet-only, real-only, 50% subset, backbone shared across 5 tasks. One honest sentence.

This is intellectually honest and reviewer-defensible. It is also *true*.

---

## 4. "Benchmarkable" thresholds per head (answering 50 Q1.3)

| Head | Current (grounded) | "Benchmarkable" bar | Source of current |
|---|---|---|---|
| Detection (assembly-state) | `mAP50_pc = 0.304` / `mAP50 = 0.202` | ≥ 0.30 pc + clean confusion matrix | `rf_stage_state.json` |
| Head pose (9-DoF) | `forward MAE = 9.13°` | ≤ 15° (well clear) + report up/position too | `rf_stage_state.json` |
| Activity (75-class) | untrained (≈ chance) | **Top-1 ≥ ~15%** (vs 1.3% chance) = "demonstrates transfer" | `metrics.jsonl` ep0 |
| PSR | `f1_at_t = 0.0` (never learned) | **f1_at_t ≥ 0.15** OR drop | `eval_metrics.json` |
| Body pose | untrained, **no GT in IndustReal** | IKEA-only PCK@0.2, or **cut** | GUIDE 7 F1 |

**Rule:** a head goes in a results table only if it beats its trivial baseline by a clear margin. Head pose and detection qualify now. Activity qualifies after rf3 *iff* Top-1 ≥ ~15%. PSR qualifies only if the go/no-go (`54`) succeeds. Body pose is IKEA-or-cut.

---

## 5. Table-by-table fill plan (the >30 placeholders, 50 §10)

Status legend: 🟢 fillable now · 🟡 fillable after a run you control · 🔴 at risk / may cut.

### IndustReal headline (`tab:industreal-headline`)
| Row | Metric | Fill source | Status |
|---|---|---|---|
| ASD mAP (b-boxed) | `det_mAP50` | rf2 full eval (running) | 🟢 ~0.20–0.25 |
| ASD mAP@0.5 present-class | `det_mAP50_pc` | rf2 full eval | 🟢 ~0.30–0.38 |
| ASD mAP@[.5:.95] | `det_mAP_50_95` | rf2 full eval | 🟢 |
| Head pose forward MAE° | `forward_angular_MAE_deg` | rf2 eval | 🟢 9.13° |
| Head pose up MAE° / position mm | `up_angular_MAE_deg`, `position_MAE_mm` | rf2 eval (already logged) | 🟢 |
| Activity Top-1 / Top-5 | `act_accuracy`, `act_top5_accuracy` | rf3 eval | 🟡 needs rf3 |
| PSR F1(±3)/F1(±5)/POS | `psr_f1_at_t`, `psr_f1_at_t5`, `psr_pos` | rf3/rf4 eval | 🔴 needs PSR to work |

### Efficiency (`tab:efficiency`, §5, `tab:complexity`)
| Metric | Fill source | Status |
|---|---|---|
| Params (M) | model param count | 🟢 now |
| GFLOPs | `thop.profile` (`evaluate.py:2881`) | 🟢 now |
| FPS batched / streaming | `eff_fps`, `eff_fps_streaming` | 🟢 now |
| Param-reduction vs N specialists | arithmetic from above | 🟢 now |

**→ The entire efficiency table is a today-task.** See `54` Phase 0.

### Ablations
| Table | Fills via | Status |
|---|---|---|
| Head contributions / single-vs-multi (`tab:abl-heads`) | `recovery_det_only` (single-task det) vs rf2 (multi-task det) on identical backbone | 🟡 **the core experiment** (`55`) |
| FiLM ladder (`tab:abl-film`) | no-FiLM vs PoseFiLM vs HeadPoseFiLM vs both, on activity | 🔴 needs activity trained |
| MTL weighting (`tab:abl-mtl`) | Kendall vs fixed weights | 🟡 from training logs (`log_var_*` already logged) |
| Backbone (`tab:abl-backbone`) | ResNet-50 vs ConvNeXt-T | 🔴 out of budget — **cut or mark future work** |
| Temporal (`tab:abl-temporal`) | TCN/ViT component ablation | 🔴 out of budget — cut |

### IKEA ASM headline (`tab:ikea-headline`)
🔴 **All rows at risk.** 2–3 GPU-days, eval pipeline untested for IKEA. **Recommend cutting to IndustReal-only** for the first submission (see §7).

---

## 6. Conclusion section draft (answering 50 Q10.3)

> We presented POPW, a single shared-backbone model that performs detection, head pose, activity, and procedure understanding for egocentric assembly in one forward pass on commodity hardware (a single RTX 3060). On IndustReal, a unified ConvNeXt-T + FPN backbone with five lightweight heads matches its single-task counterpart on assembly-state detection within [Δ from Ablation A], while reducing parameters and inference cost by [factor] relative to an equivalent set of task-specialist models. We find that head-pose regression is learned cleanly (9.1° forward angular error) and shares the backbone without measurable interference, whereas fine-grained assembly-**state** detection — distinguishing 11-bit states that differ by a single component — is the hardest and most interference-sensitive task, plateauing well below object-detection baselines trained with large-scale synthetic data. Our ablations show that [FiLM conditioning contributes / is inconclusive], and that Kendall-weighted staged training prevents the catastrophic collapse otherwise observed when all heads are activated simultaneously. POPW is not state-of-the-art on any single task; its contribution is to quantify, on accessible hardware, what is gained (efficiency, a single deployable model) and lost (fine-grained detection accuracy) by unifying egocentric assembly understanding. Limitations include single-GPU training budgets, a 50% data subset for the shared stage, RGB-only activity recognition (vs multi-modal baselines), and procedure-step recognition that we report as [working at F1=… / a negative result requiring further work]. Future work: full-data and synthetic-pretrained detection, the IKEA-ASM second domain, and multi-seed variance.

Fill the four `[brackets]` from the actual runs. Note it **states the negative results plainly** — that is what makes it credible.

---

## 7. What to cut, in order (answering 50 Q12.3, R-series)

1. **IKEA ASM** (and with it, **body pose**) — 2–3 days, untested pipeline (`55` R5). Reframe as IndustReal-only egocentric paper. *Cut first.*
2. **Backbone + temporal ablations** — out of budget. Mark as future work; the FiLM + single-vs-multi ablations carry the scientific claim.
3. **PSR** — if the go/no-go (`54`) fails, drop to a one-paragraph negative result.
4. **Ablation B (FiLM)** — if activity is too weak to modulate, report as inconclusive, keep the architecture description.
5. **Multi-seed** — single seed for submission; promise 3-seed for camera-ready (standard and acceptable, 50 R7).

**Irreducible publishable core (workshop-floor):** detection (honest, reframed) + head pose + efficiency + **Ablation A on detection** + honest limitations. Everything else is upside.

---

## 8. Figures: what's free vs new code (50 Q10.4)

| Figure | Status |
|---|---|
| 24×24 detection confusion matrix | 🟢 already saved (`evaluate.py` P5) |
| Activity confusion matrix (75×75) | 🟢 already computed (`act_confusion_matrix` in eval) |
| Kendall weight evolution | 🟢 `log_var_det/pose/act/psr` already in `metrics.jsonl` — just plot |
| Per-task val curves | 🟢 from `metrics.jsonl` |
| Architecture diagram | 🔴 manual (draw.io) — half a day |
| Qualitative 3×3 grid | 🟡 needs an inference-overlay script (~2h) |

Four of six figures are **plot-from-existing-logs**. Prioritize those.
