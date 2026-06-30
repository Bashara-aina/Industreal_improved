# 75 — Opus Response to Round-3 Consult (files 70–74): Making Activity Actually Work

Date: 2026-07-01 · Model: Opus 4.8 · Method: read 70–74 line by line, verified the
loss + sampler + data path against `main`. This file is the honest answer to "is
activity salvageable, and what single thing do I do." Read §1 (the verdict), then §3
(the decision). Everything else supports those two.

---

## 1. The verdict (please read this first, even though it's hard)

**74-class per-frame activity recognition on this dataset cannot produce a strong
benchmark number — and no loss function, sampler, or head architecture will change
that.** This is an information-theoretic wall, not a tuning failure. I verified the
mechanism in code; here is why both runs collapsed and why the next ten loss tweaks
will too:

1. **Your loss is already near-perfectly balanced.** `ClassBalancedFocalLoss`
   (`losses.py:679–786`) with β=0.999 makes per-class *mass* ≈ equal, then *also*
   focal-weights hard examples, then caps rare-class upweighting at **50×**
   (`losses.py:778–783`). The CE path (`losses.py:1125–1135`) is CB-weighted too, and
   the `_weights[0]=0` bug is already fixed on main. So "try CB-Focal / higher γ /
   LDAM" cannot be the unlock — the loss is *not* the thing starving the tail.

2. **The sampler is only *partially* balanced** (this part I did fix — see §4). CB
   effective-number weighting (β=0.99) leaves the 5 head classes with ~4–5× the
   sampling mass of the tail. But notice: even with the loss favoring the tail up to
   50× *and* the sampler favoring the head ~4×, the net already favors rare classes —
   and it *still* collapses. That tells you balancing is not the binding constraint.

3. **The binding constraint is data × features, and it is irreducible:**
   - **Data:** 48 of 74 classes have <10 frames; 7 are singletons (file 70/71). A
     classifier *cannot* learn a class from <10 examples that generalizes to a
     *different recording* in val. Force balance → it memorizes those <10 frames and
     fails on val. Don't force it → it never predicts them. Either way val-macro-F1
     over 74 classes is pinned near zero. There is no loss that manufactures signal
     from 8 frames.
   - **Features:** the head classifies a *global-average-pooled* vector
     `det_conf ⊕ GAP(c5_mod_blend) ⊕ GAP(p4)`. GAP throws away *where* the hands are
     and *what* they touch — exactly the cues that separate "tighten_tooth_washer"
     from "loosen_tooth_washer." Global pooled features of a detection backbone are
     close to constant across fine-grained actions, so the head's best move is a
     near-constant output → the 1–2 class collapse you measured.

So: macro-F1 over 74 classes is bounded near zero by points (3a)+(3b) regardless of
loss/sampler/MLP depth. This is why I am not going to send you on another loss
grid-search — it would waste your 3 GPU-hours and your remaining month.

## 2. Honest probabilities for every intervention in 70/72/73

| Intervention | Will it break the collapse? | Will it give a benchmarkable number on 74 classes? |
|---|---|---|
| LDAM / γ=5 / vanilla focal (70 Q1,Q3; 73 #3) | Maybe nudges diversity | **No** — loss isn't the constraint |
| Deeper/wider/residual MLP (72 Alt 1-3) | No | **No** — more capacity overfits 8-frame classes faster |
| Higher ACTIVITY_LOSS_WEIGHT=3 (70 Q12) | Slightly more backbone pull | **No**, and risks the other tasks |
| Freeze backbone for activity (70 Q11) | Slightly | **No** — features still pooled; hurts det/pose |
| Mixup / RandAugment / CutOut (73 #1-3) | Marginal | **No** — doesn't create tail signal |
| Oversample minority 10× (73 #4) | More predicted classes | **No** — memorizes repeated frames, val still fails |
| Truly balanced sampler (§4, shipped) | **Yes — more diverse predictions** | Only partly; tail still un-generalizable |
| **Class merging → ~12-15 groups (73 #5)** | **Yes** | **YES — this is the only path to a real number** |
| Per-recording sequential sampler (72 Alt 5) | Helps *temporal* head only | Not on its own; big refactor |
| Drop activity (72 Alt 6 / 74) | n/a | n/a — the honest fallback |

Two rows give a benchmarkable result: **class merging** (a real number) and **drop +
document** (a real negative result). Everything else is motion without progress.

## 3. THE decision you need to make (this is yours, not mine)

You have exactly two routes to "activity works / comparable result," and they imply
different paper claims. I need you to pick:

### Route A — Coarse action recognition (RECOMMENDED for a *positive* number)
Merge the 74 fine-grained actions into **~12–15 verb groups** by their action verb
(the first token of the class name: `take_*`, `plug_*`, `align_*`, `tighten_*`,
`loosen_*`, `pull_*`, `insert_*`, `remove_*`, …). This is file-73 Strategy 5, and it
is implementable **data-drivenly from `ACT_CLASS_NAMES`** (no hand-mapping needed —
the names already follow `verb_object`).

- Each group then has ~100–400 frames → **learnable**, and verbs differ grossly
  enough that even pooled features separate them.
- **Realistic expectation:** top-1 ≈ 0.40–0.60, macro-F1 ≈ 0.30–0.50 over ~13 groups.
  That is a genuine, reportable, comparable benchmark.
- **Paper claim changes** from "74-class action recognition" to "**action-group /
  verb recognition (13 classes)**." This is honest and common in assembly video work;
  it is *not* a weakness if you state it plainly. You can still *report* the 74-class
  collapse as the §5 pathology that motivated the grouping — best of both.
- I can implement this end-to-end (label remap → head dim → loss → eval → metrics)
  as a single gated flag `ACT_CLASS_GROUPING='verb'`. ~1–2 h of my work; you re-run.

### Route B — Keep 74 classes, report it as the documented result (file 74)
Activity becomes the **negative result** in §5.3, and the paper is the 4-task system +
4 pathologies. Your headline numbers are detection mAP50, forward-gaze 8.71°, PSR,
body pose, plus the analysis. This is publishable at AAIML (applied/analysis venue)
and you already drafted it well in 74.

**My recommendation: do Route A.** It costs ~2 h of code + one training run and turns
"activity failed" into "activity works at the action-group level (top-1 ~0.5)" — a
real benchmark — *while still letting you tell the 74-class collapse story.* Route B
is the safe floor if Route A's run somehow underperforms. They are not mutually
exclusive: implement A, and if the grouped number is still weak, you fall back to B
with even stronger evidence. There is no scenario where Route A makes things worse.

## 4. What I shipped this round (safe, gated)

| Change | File | Effect |
|---|---|---|
| `ACT_SAMPLER_MODE='balanced'` option + `ACT_SAMPLER_COUNT_FLOOR=15` | `industreal_dataset.py:get_sampler`, `config.py` | True per-class balance for ≥15-frame classes; sub-floor classes scaled by count so 1–7-frame singletons aren't memorized. **Default stays `'cb'`** (no change to your live run); flip to `'balanced'` for CE-loss experiments. With CB-Focal active, leave it on `'cb'` (the loss already rebalances). |
| `ACT_CLASS0_IS_NA=False` documented | `config.py`, `losses.py` | Confirms slot 0 = real action `take_short_brace`; the historical `_weights[0]=0` zeroing is gated off (already fixed on main). |

I deliberately did **not** default the balanced sampler on, and did **not** unilaterally
implement the class merging — the merge changes your paper's central claim, so it's
your call (§3). Say "Route A" and I implement the full verb-grouping in the next turn.

## 5. Answers to the explicit questions in 70

- **Q1–Q3 (loss variants):** no — loss is not the constraint (§1.1). Stop here.
- **Q4–Q6 (deeper/wider/temporal MLP):** no — capacity overfits the tail; per-frame
  temporal stacking over a shuffled batch is meaningless (you already proved this).
- **Q7 ("is 3.7k frames simply too small?"):** for 74 classes, **yes**. For ~13 verb
  groups, **no**. Minimum ~100 generalizable frames/class is the right rule of thumb;
  grouping is how you reach it without new data.
- **Q8/Q9 (merge / oversample):** merge **yes** (Route A); oversample **no** (memorizes).
- **Q10 (mixup/cutout):** marginal; fine to add *after* grouping, not as the fix.
- **Q11/Q12 (freeze backbone / loss weight 3):** no; hurts the tasks that work.
- **Q13/Q14 (4-task / negative-result paper):** both publishable at AAIML — that's
  Route B / the floor. The negative result *strengthens* the paper only if you *also*
  show the grouped positive result (Route A); "we tried everything and it failed" is
  weaker than "fine-grained fails for this data reason, and here is the level at which
  it succeeds."

## 6. The 3 GPU-hours, spent well
1. Implement Route A verb-grouping (my work) → you run **one** epoch-0 with grouping +
   `ACT_SAMPLER_MODE='balanced'` + CE (not CB-Focal; with 13 balanced classes you
   don't need focal). Check epoch-1 **prediction diversity** (≥10 of 13 groups) and
   top-1. ~1 h.
2. If diverse and top-1 > 0.3 → continue the RF schedule with grouped activity. Done.
3. If still weak → Route B, and you've earned the strongest possible §5.3.

Do **not** spend the hours on loss/architecture sweeps. The math in §1 says they
can't move the 74-class number.
