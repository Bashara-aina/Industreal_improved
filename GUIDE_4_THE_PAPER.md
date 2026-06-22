# GUIDE 4 — THE PAPER (finish it, prove the idea)

*How to frame your contribution so it is **proven** with the numbers GUIDE_3 says are
achievable — not the SOTA numbers you've been chasing. Your idea is provable today.*

> "I need my idea is working. That is the main point — I finished my paper and my idea is
> proven." — This guide is built around that sentence.

---

## 1. What your idea actually is (and what proves it)

Your thesis is **not** "we beat YOLOv8m / MViTv2 / STORM-PSR." Your thesis is:

> **A single shared-backbone model can perform egocentric assembly understanding —
> detection, body pose, head pose, activity, and procedure-step recognition — in one
> forward pass, at a fraction of the parameters and compute of separate specialists,
> *without catastrophic interference*, with cross-task FiLM conditioning.**

That thesis is proven by **four** things, all achievable now:

1. **All five heads produce non-trivial, competitive results** (GUIDE_3 targets). ← "it works"
2. **Efficiency:** 53M params / 1 forward pass vs ~81M / 3. ← measured, uncontested
3. **The multi-task model is competitive with single-task specialists** (your ablation). ← "sharing doesn't break it / helps"
4. **Two-stage FiLM conditioning contributes** (your ablation). ← the novel mechanism

You need **zero** SOTA wins to prove this. You need *all heads alive + the two ablations*.

---

## 2. The ONE experiment that proves the idea (do not skip this)

Everything rides on this 2×N table. It is the scientific core of the paper.

**Ablation A — does multi-task sharing work?**

| Configuration | Det | Act | PSR | HeadPose | Params | Passes |
|---------------|-----|-----|-----|----------|--------|--------|
| Single-task specialists (your heads, trained alone) | … | … | … | … | sum | N |
| **POPW unified (shared backbone, all heads)** | … | … | … | … | **53M** | **1** |

The publishable finding is **any** of these three (you don't get to choose — you report
what you measure, and all three are interesting):
- **Multi-task ≈ specialists** → "shared backbone matches specialists at 1/3 compute." ✅ strong
- **Multi-task > specialists on some heads** → "cross-task transfer helps." ✅ strongest
- **Multi-task < specialists but close** → "modest interference; efficiency trade-off quantified." ✅ still a real, honest contribution

The decoupled training (GUIDE_2) makes this ablation *cheap and clean*: Phase-A backbone =
the shared trunk; train each head alone (single-task) vs all heads on the shared cache
(multi-task). Same backbone, same data, same eval. Clean comparison.

**Ablation B — does FiLM conditioning help?**

| Configuration | Activity Top-1 | Notes |
|---------------|----------------|-------|
| No conditioning | … | backbone features only |
| + PoseFiLM | … | body keypoints → γ,β |
| + PoseFiLM + HeadPoseFiLM | … | the full two-stage design |

If FiLM helps activity even by 1–3 points, your "cross-task information flow" novelty is
demonstrated. If it's neutral, report it honestly and lean on the efficiency + unified-
architecture contributions (still a complete paper).

---

## 3. How to present detection honestly (turn the weakness into a result)

Detection is your hardest head. **Do not hide it or apologize — frame it as a finding:**

1. Report `det_mAP50_pc` (~0.33–0.45) as the honest number, with `det_mAP50` + n_present
   for comparability (GUIDE_3 §2.1).
2. Show the **24×24 confusion matrix**: errors concentrate on **1-bit-Hamming-neighbor
   assembly states**. Write the sentence: *"The residual error is dominated by confusion
   between assembly states differing by a single component — i.e., the task is
   fine-grained state discrimination, not localization (localization recall is high;
   see Table X)."* This is a genuinely interesting characterization of the IndustReal ASD
   task and elevates your paper above a leaderboard entry.
3. State the synthetic-data gap explicitly (next section). The reviewer who knows YOLOv8m
   used 260k synthetic images will respect the honesty and *expect* the gap.

---

## 4. The limitations section (write it proactively — it disarms reviewers)

A strong limitations section makes weak numbers defensible. Include:

1. **No synthetic pretraining for detection.** "The YOLOv8m baseline (0.838) used COCO
   pretraining + 260k synthetic images + real fine-tuning. POPW detection is trained on
   real data only; we estimate synthetic pretraining would close much of the gap (our
   `pretrain_synthetic` path is implemented; running it at scale is future work)." *(If
   you DO have the synth data, run it — GUIDE_5 — and this limitation shrinks to a win.)*
2. **Single 12 GB GPU constraint** shaped design choices (batch size, RGB-only activity vs
   a K400 video encoder). Frame as *accessibility*, not weakness — "POPW trains on
   commodity hardware."
3. **Fine-grained ASD ceiling** from near-identical visual states + class imbalance
   (some states < 100 instances). Quantified by the confusion matrix.
4. **Subset training during development**, full-test evaluation for final numbers.

---

## 5. Filling the paper (map numbers → `.tex` placeholders)

`popw_paper_improved.tex` has `\popwres`/`\todo` placeholders (~lines 539–625). Fill them
from the GUIDE_3 §4 scripts:

| Paper table/row | Source metric | From |
|---|---|---|
| ASD detection | `det_mAP50_pc`, `det_mAP50` (+n_present), mAP@[.5:.95] | full-test `evaluate.py` + `diag_per_class_truth.py` |
| ASD confusion fig | 24×24 confusion | `evaluate.py` (add confusion dump if needed) |
| Activity | clip Top-1/Top-5 | full-test `evaluate.py` (clip-level) |
| PSR | F1(±3), POS | full-test `evaluate.py` |
| Head pose | fwd/up/pos MAE | full-test `evaluate.py` |
| Body pose | PCK@0.2, MAE | full-test `evaluate.py` |
| Efficiency | params, GFLOPs, FPS | `efficiency_report.py` |
| Ablation A (MTL) | per-task, single vs unified | Phase-A/B runs |
| Ablation B (FiLM) | activity Top-1 ladder | 3 short Phase-B runs |

**Do not block the paper on detection.** Fill every other row first; detection is one row.

---

## 6. The narrative arc (slides + paper intro, reusable)

1. **Problem:** assembly understanding today needs ~3–5 separate models (YOLOv8m, MViTv2,
   STORM-PSR, pose nets). Redundant compute, no shared representation.
2. **Idea:** one shared backbone + 5 task heads + two-stage FiLM conditioning, one forward
   pass.
3. **Challenge & method:** naive joint training causes gradient interference on commodity
   hardware; we use a decoupled backbone→frozen-cache→heads schedule (+ optional joint
   fine-tune) that makes all heads learn stably.
4. **Results:** all five heads competitive; uncontested head-pose result; 35% fewer params
   / 1-vs-3 passes; MTL matches specialists (Ablation A); FiLM helps (Ablation B).
5. **Honesty:** detection is fine-grained state ID (confusion matrix), real-data-only;
   synthetic pretraining is future work.
6. **Takeaway:** unified egocentric assembly understanding is feasible and efficient on a
   single GPU — *the idea works.*

---

## 7. Reviewer rebuttal prep (anticipate the 4 obvious attacks)

| Likely reviewer comment | Your prepared answer |
|---|---|
| "Detection is far below YOLOv8m." | Different data budget (no 260k synth); we report honest present-class mAP; residual error is fine-grained state confusion (Fig X); efficiency + unified design are the contribution. |
| "Activity below MViTv2." | RGB-only, no K400 video encoder, multi-task shared backbone vs single-task specialist; clip-level protocol matched; competitive at fraction of compute. |
| "Is multi-task actually helping?" | Ablation A quantifies it head-by-head vs single-task on the identical backbone; we report the honest delta (neutral/positive). |
| "Why not just use 3 specialists?" | Table: 53M/1-pass vs 81M/3-pass; FPS on commodity GPU; that *is* the point. |

---

## 8. Definition of "paper done"

- [ ] All five heads have a non-trivial number on the **full test set** (GUIDE_3 targets).
- [ ] Efficiency table filled (params/FLOPs/FPS).
- [ ] Ablation A (single vs multi-task) filled.
- [ ] Ablation B (FiLM) filled.
- [ ] Detection confusion-matrix figure + honest framing.
- [ ] Limitations section written.
- [ ] Multi-seed (≥ headline tasks) mean±std.
- [ ] Every `\todo`/`\popwres` in the `.tex` replaced.

When these boxes are checked, **the idea is proven and the paper is done.** That is the
finish line — not mAP 0.40.

➡ **Next:** GUIDE_5 — the exact commands and day-by-day plan to execute all of this.
