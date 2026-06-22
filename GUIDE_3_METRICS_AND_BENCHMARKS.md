# GUIDE 3 — METRICS & BENCHMARKS (measure honestly, prove it)

*How to evaluate each head correctly, what numbers count as "good enough," and how to turn
them into the paper's results tables. Wrong metrics have cost you more than wrong training.*

> Prereq: GUIDE_1 (why the metric mattered). Training: GUIDE_2. Commands: GUIDE_5.

---

## 1. The golden rule

**Report the metric the baseline reported, computed the way the baseline computed it.**
Every points-leak you've suffered (the −40% dilution, the per-frame-vs-clip activity gap)
came from a protocol mismatch. Fix the protocol once, here, and never argue with the number
again.

---

## 2. Per-task evaluation protocol

### 2.1 Detection (ASD) — present-class mAP, and be explicit
- **Primary (honest):** `det_mAP50_pc` — mean AP@0.5 over channels with GT>0 only. This is
  now what your `best.pth` and gates use (I changed it). **~0.33–0.45 is your real number.**
- **Also report (for comparability):** `det_mAP50` (COCO-24 mean) — but *always with*
  `n_present`/24 stated, so reviewers know ~8 channels are empty in the subset. On the
  **full** test set more classes are present, so the gap shrinks; run final eval on the
  full test split, not the 50% subset.
- **mAP@[0.5:0.95]:** report it; it'll be lower (fine-grained boxes), that's expected.
- **Add a 24×24 confusion matrix.** It will show errors concentrate on 1-bit-Hamming
  neighbor states. This single figure *reframes* your "low mAP" as "fine-grained state
  confusion," which is a scientifically interesting result, not a failure (GUIDE_4 §3).
- **Anchors are fine** (POS_ANCHOR_PROBE proved 400–800 pos/image). Do not re-litigate.

### 2.2 Activity — CLIP-LEVEL Top-1/Top-5 (this is free points)
- Baseline (MViTv2) evaluates **clip-level**: 16 uniformly-sampled frames per action
  segment → **one** prediction per segment. You have been evaluating **per-frame**, which
  costs double digits.
- Aggregate per clip: average logits (or majority vote) over the segment's frames, then
  argmax once. Report **clip Top-1 and Top-5**. Keep per-frame as a secondary diagnostic.
- Target: **clip Top-1 0.35–0.45, Top-5 0.70+.** That is "competitive with MViTv2 at a
  fraction of compute, RGB-only, no K400 video encoder" — a legitimate claim.

### 2.3 PSR — F1(±3 frame) and POS, transition-based
- Baselines: B2 F1=0.731/POS=0.816 (heuristic), STORM-PSR F1=0.506/POS=0.812.
- Evaluate the **±3-frame F1** (the benchmark's tolerance window) and POS. Your
  `psr_f1_at_t` is the right metric (NOT `psr_overall_f1`, which is ~0 for all-ones).
- Decode with the **monotonic + procedure-order** constraints (you have them). Report F1
  and POS. Target **F1 0.50–0.62** → "beats STORM-PSR's neural F1, approaches the B2
  heuristic, with a *learned* model."

### 2.4 Head pose — geodesic/angular MAE (uncontested win)
- Report forward-angular MAE, up-angular MAE, position MAE (mm). You're at **~9°** forward
  — excellent. **No published supervised baseline exists**, so this is a clean
  contribution. State the 6D-rotation parameterization (Zhou et al. 2019) as the reason.

### 2.5 Body pose — PCK@0.2 + MAE
- Standard keypoint metrics. Report PCK@0.2 and per-keypoint MAE. Secondary head; just
  show it's alive and reasonable.

### 2.6 Efficiency — your safest win, quantify it precisely
- **Params:** ~53M (POPW) vs ~81M (YOLOv8m 26M + MViTv2 35M + STORM-PSR ~20M).
- **Forward passes:** 1 vs 3.
- **GFLOPs** at 1280×720 and **FPS** on the RTX 3060 (use your `efficiency_report.py`).
- This table needs *no accuracy at all* to be true. Make it prominent.

---

## 3. The target table, with justification ("good enough" = these)

| Task | Metric | Target | Why this is "good enough" |
|------|--------|--------|---------------------------|
| Detection | `det_mAP50_pc` | 0.33–0.45 | Real-data-only, unified, no synth pretrain. YOLOv8m's 0.838 bought with 260k synth — stated as limitation. |
| Detection | `det_mAP50` (full test) | report w/ n_present | Comparability; honest about dilution. |
| Activity | clip Top-1 / Top-5 | 0.35–0.45 / 0.70+ | RGB-only, no K400 encoder, multi-task. MViTv2 0.6525 is single-task K400. |
| PSR | F1(±3) / POS | 0.50–0.62 / 0.75+ | Beats STORM-PSR neural (0.506), learned (not heuristic like B2). |
| Head pose | fwd MAE | ≤ 15° (have 9°) | No baseline — uncontested. |
| Body pose | PCK@0.2 | report | Completeness. |
| Efficiency | params/FLOPs/FPS | 53M / 1 pass | 35% fewer params, 1 vs 3 passes — by construction. |

**You do not need a single SOTA number.** Two uncontested wins (head pose, efficiency) +
three honest competitive results = a complete paper. See GUIDE_4 for how to say it.

---

## 4. How to generate the final tables (scripts you already have)

- **Per-class detection truth:** run `diag_per_class_truth.py` (Opus v11 added it) on your
  run — gives the authoritative per-class AP/GT table and `det_mAP50_pc`. This kills index
  ambiguity and feeds the detection table + confusion-matrix figure.
- **Full evaluation:** `src/evaluation/evaluate.py` on the **full test split** (not the
  subset) for the final numbers. The honest metric keys (`det_mAP50`, `det_mAP50_pc`,
  `det_n_present_classes`, `act_clip_accuracy`, `act_top5_accuracy`, `psr_f1_at_t`,
  `psr_pos`, `forward_angular_MAE_deg`) are all already produced.
- **Efficiency:** `scripts/training/efficiency_report.py` and
  `scripts/training/generate_paper_tables.py`.

Map each number to the `\popwres`/`\todo` placeholders in `popw_paper_improved.tex`
(lines ~539–625). GUIDE_4 §5 has the placeholder→metric mapping.

---

## 5. Rigor the reviewers will ask for (do these, they're cheap)

1. **Full test split, not the 50% subset**, for the headline table. The subset was for
   fast iteration; final numbers must be on the official test set.
2. **Multi-seed** (42, 123, 7) for at least the two headline tasks; report mean ± std.
   Decoupled Phase-B heads train in minutes, so 3 seeds is cheap there.
3. **One ablation that proves the idea** (GUIDE_4 §2): single-task vs multi-task, and
   with/without FiLM. This is the scientific core — budget for it.
4. **Same eval protocol as each baseline** (clip-level activity, ±3-frame PSR). State the
   protocol explicitly in the paper so the comparison is defensible.

---

## 6. Common metric traps (you have hit several — don't again)

| Trap | Symptom | Fix |
|------|---------|-----|
| Diluted mAP | "stuck at 0.207" | use `det_mAP50_pc` (done in code) |
| AP=1.0 on empty class | "architecture learns perfectly!" | it means *no GT* — exclude/flag it |
| Per-frame activity eval | Top-1 looks terrible | clip-level aggregation |
| `psr_overall_f1` ≈ 0 | "PSR dead" | use `psr_f1_at_t` (±3-frame) |
| Combined metric reads "almost 0.50" | false hope | it's MAE-dominated; judge per-task (GUIDE_1) |
| Subset val, rare class AP=0 | "class 6 bug" | data scarcity (65 train) — report, don't grind |
| Grad-norm head vs backbone | "detection suppressed 140×" | norm scales with param count; compare head-to-head |

➡ **Next:** GUIDE_4 — how to frame these numbers so the paper *proves your idea*.
