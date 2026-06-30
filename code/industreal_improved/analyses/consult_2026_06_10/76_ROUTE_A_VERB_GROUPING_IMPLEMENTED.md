# 76 — Route A Implemented: Verb-Grouping for Activity [2026-07-01]

Implements the Route-A decision from file 75: collapse the 74 fine-grained actions
into ~12–15 **verb groups** so the activity task becomes learnable and produces a real,
benchmarkable number. Fully gated; **default OFF** — your current pipeline is unchanged
until you flip one flag.

## How to turn it on

1. Inspect the real grouping first (no training, ~1 min):
   ```bash
   python scripts/verify_act_grouping.py
   ```
   This prints the verb groups, the raw-id→group map, and per-group TRAIN frame counts.
   Confirm each group clears ~100 frames (the script flags any that don't).
2. Activate in `config.py`:
   ```python
   ACT_CLASS_GROUPING = 'verb'      # was 'none'
   ```
3. Recommended loss/sampler for the grouped run (13 balanced classes don't need focal):
   ```python
   USE_CB_FOCAL_ACT = False         # plain CE+label-smoothing is enough at 13 classes
   ACT_SAMPLER_MODE = 'balanced'    # the file-75 balanced sampler (now safe: no focal double-emphasis)
   ```
   (If you prefer to change nothing else, leaving CB-Focal on also works — it just
   over-emphasizes rare groups slightly.)
4. Start a **fresh** RF4 run with `--reinit-heads` (the head changes width: 75 → ~13).
   A checkpoint with a 75-wide activity classifier will NOT load into a 13-wide head,
   so you must reinit, not resume.

## Go/no-go (epoch 1, same as file 69 Decision 2)
- `#distinct predicted groups ≥ 10` of ~13, AND `entropy ≥ ~1.5 nats`, AND
  `act_macro_f1 > 0.05`.
- Realistic target once trained: **top-1 ≈ 0.40–0.60, macro-F1 ≈ 0.30–0.50** over the
  verb groups. That is your benchmarkable activity result.

## What changed (4 production sites + config + 1 script)

| File | Change |
|------|--------|
| `src/config.py` | `ACT_CLASS_GROUPING`, `_build_act_grouping()`, `ACT_ID_TO_GROUP`, `ACT_GROUP_NAMES`, `NUM_ACT_GROUPS`, `NUM_ACT_OUTPUTS`, `ACT_OUTPUT_NAMES`, `remap_activity_label()`. Identity when `'none'`. |
| `src/data/industreal_dataset.py` | Remap each frame's `action_label` (per-frame **and** sequence path) through `remap_activity_label`; `activity_ids` + `class_counts` built in group space (`minlength=NUM_ACT_OUTPUTS`). `-1` preserved. |
| `src/models/model.py` | `ActivityHead` width = `NUM_ACT_OUTPUTS`. |
| `src/training/train.py`, `src/evaluation/evaluate.py` | `MultiTaskLoss(num_classes_act=NUM_ACT_OUTPUTS)`; eval class names = `ACT_OUTPUT_NAMES`; diversity `num_cls` = `NUM_ACT_OUTPUTS`. |
| `scripts/verify_act_grouping.py` | Pre-flight inspector (real names + per-group counts). |

`NUM_CLASSES_ACT` stays 75 (raw label space / asserts untouched). Everything that sizes
the *head, loss, sampler, and eval* now reads `NUM_ACT_OUTPUTS`, which equals 75 when
grouping is off and ~13 when on. Verified: all six modified files `py_compile` clean;
the grouping logic passes a standalone unit test (NA/take/plug/align/tighten/loosen/
pull/insert/remove/check/put/unknown → 12 groups on a representative name set; `-1`
preserved; identity mode → 75).

## One caveat to check (cosmetic, not headline)
The eval's `act_accuracy_no_na` excludes group **0**, assuming index 0 = "NA". If your
real `ACT_CLASS_NAMES[0]` is the action `take_short_brace` (per file 70), group 0 is the
`take` verb, so that one *secondary* metric would exclude a real group. **The headline
metrics (top-1, macro-F1 via the present-label filter) are unaffected.** `verify_act_grouping.py`
shows you `names[0]` so you can confirm; if it's a real action, just report top-1 /
macro-F1 (not `_no_na`) for the grouped task, or I can add an `ACT_NA_GROUP_INDEX`
guard — say the word.

## Why this is the right call (recap from 75)
74-class per-frame AR is bounded near zero by data (48/74 classes <10 frames) and
global-pooled features — no loss/sampler/architecture fixes it. Verb groups have
~100–400 frames each and differ grossly enough to be separable from pooled features.
You still report the 74-class collapse as the §5 training-pathology contribution; the
grouped result is the positive number. Route B (4-task, activity-as-failure) remains the
fallback if the grouped run underperforms — but it cannot do worse than the current
collapse.
