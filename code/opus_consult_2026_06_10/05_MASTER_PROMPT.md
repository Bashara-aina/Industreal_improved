# The Master Prompt — End-to-End Request to Opus

> **READ THIS FILE FIRST.** This is the prompt you (Opus) should
> answer. Everything else in this folder is evidence and context
> for this prompt. You are not required to read the other MDs
> before answering, but you should skim `02_COLLAPSE_CRISIS.md`
> and `04_HYPOTHESES_FOR_OPUS.md` because they contain the
> diagnostic outputs and the specific questions.
>
> Your job: produce **5+ implementable MD files** that we can drop
> into our codebase and run within the next 7 days. Each MD file
> is a concrete implementation plan for a specific change, with
> code, expected metrics, and rollback procedure.

---

## 1. The Project, In One Paragraph

We are building **POPW** (Pose-conditioned multi-task worker), a
unified computer-vision model that takes a single RGB frame from
an egocentric toy-assembly video and produces **five outputs
simultaneously**: 24-class object detection (ASD), 17-keypoint
body pose, 9-DoF head pose, 74-class activity recognition, and
11-binary procedure-step recognition (PSR). The current
architecture shares a ConvNeXt-Tiny + FPN backbone across all
five heads and uses FiLM conditioning to inject pose information
into the activity branch. The dataset is IndustReal (a
re-distribution of the public release, 17 training recordings,
4 validation recordings at our 5% subset). Hardware: a single
NVIDIA RTX 3060 with 12 GB VRAM.

The reference paper (`popw_paper_improved.tex`) is the design we
are trying to beat, and the closest published baselines per task
are:

| Task | Baseline | Metric | Our current |
|---|---|---|---|
| Detection | YOLOv8m | mAP@0.5 = 83.80% | **0.00%** |
| Activity | MViTv2 (RGB only) | Top-1 = 65.25% | **0.00%** (Top-5) |
| PSR | PSRT-B2 / STORM | F1 = 0.731 / 0.506 | **0.0000** |
| PSR | PSRT-B2 / STORM | POS = 0.816 / 0.812 | **0.0000** |
| Body pose | not reported | not reported | MAE = 739 mm (broken scale) |
| Head pose | not reported | not reported | Angular MAE = 61° (non-zero) |

(The "current" numbers are from
`evidence/metrics_baseline_pre_reinit.json`, which was produced
by loading our 39-epoch checkpoint. Three of five heads are
dead; the metric is zero because the model has collapsed, not
because the eval is broken.)

## 2. What Just Happened (the Collapse, in 60 seconds)

We trained for 39 epochs on a 5% subset. Stages 1 and 2 (detection
only, then detection+pose) trained cleanly. At the start of
stage 3 (activity and PSR come online), three of the five heads
collapsed within 5–10 batches:

- **Detection** cls_score saturated to ~1.0 on every anchor.
  64,950 predictions across 42 GT boxes (1546× ratio). mAP@0.5 = 0.
- **Activity** converged to predicting class 27 (the most common
  class) on 100% of frames. Per-class accuracy = 0% for every
  one of the 7 classes that had GT samples in the val set.
- **PSR** stuck at all-zeros across all 11 binary outputs. Sigmoid
  was 0.27 (bias = -1.0, below the 0.5 threshold). The model had
  no reason to push the bias up because the loss was dominated
  by the 0-class.

Body pose and head pose **survived**. The 7th head that mostly-
worked (head pose) is the only signal we have that the
architecture can train end-to-end on this hardware.

The NaN cascade that triggered the crash was a side effect of
**bf16 AMP** combined with the **PSRs sequence-mode** batch: the
causal transformer's 4×256 attention matrix in bf16 has
underflow in the softmax denominator, the autograd graph is
retained across the next non-seq batch, and NaN appears in PSR
loss and propagates through the multi-task aggregation.

**Full evidence:** see `02_COLLAPSE_CRISIS.md`. **Verbatim
log lines:** see `logs/train_digest.log` (header, reinit block,
tail). **The dead checkpoint's eval output:** see
`evidence/metrics_baseline_pre_reinit.json`.

## 3. What We Just Did (the Recovery, in 60 seconds)

Four surgical fixes (`03_CURRENT_RECOVERY.md`):

1. **Reinit 3 dead heads** with healthier biases
   (`train.py:1565-1720`, the `--reinit-heads` flag):
   det.pi=0.05 (was 0.01), act.bias=-0.5 (was 0), psr.bias=-0.2
   (was -1.0). Backbone and 2 working heads are NOT touched.
2. **FP32** (`config.py:289`, `MIXED_PRECISION=False`): 2× VRAM
   cost, 2× time cost, zero NaN. Batch=4 OOMs, auto-reduces
   to batch=2.
3. **Detach seq-mode PSR output** (`train.py:~1660`): breaks
   the autograd graph poisoning. The seq-mode batch still
   runs but its output doesn't flow back to non-seq weights.
4. **Hard-clamp Kendall log_vars to [-4, 2]**
   (`train.py:1782`, `_clamp_kendall_log_vars`): stops the
   model from silently zeroing a head's gradient by inflating
   its variance.

The retrain (PID 2416305, 5 epochs, FP32) is running. As of 8
minutes in: loss dropped from 90 to 20, all 5 head losses are
active and varying. The retrain is expected to finish at
~11:50, and we will re-eval with
`run_eval_post_retrain_fp32.sh`.

**The 4 fixes are a stabilization, not a solution.** They
paper over the architecture's structural problems. The absolute
metrics after the retrain will be far below the paper's
baselines. The next step is architectural changes.

## 4. The Hard Constraints (binding)

These are **non-negotiable** for any solution you propose:

- **Hardware:** single RTX 3060, 12 GB VRAM. We cannot move to
  a 24 GB card. Any solution must fit in 12 GB at inference
  time and at training time (training batch=2 in FP32 is the
  current best).
- **Dataset:** IndustReal, 17 training recordings, 4 validation
  recordings at our 5% subset (`SUBSET_RATIO=0.05`). We can
  scale to 10% or 20% subset, but not to the full 17-recording
  set, because each recording is ~2 hours of video at 30 fps.
- **Inference speed:** the model must produce a per-frame
  prediction in < 200 ms on the 3060. The current architecture
  is at < 500 ms per batch of 4, so per-frame is ~125 ms. We
  have headroom, but not unlimited.
- **Eval framework:** the existing `evaluate.py` produces the
  metrics. We will judge success on the metrics it produces.
  You can add new metrics, but you cannot remove the existing
  ones without a strong reason.
- **Code style:** Python, PyTorch 2.x, no breaking changes to
  the dataset API (`industreal_dataset.py`). Anything that
  requires changing the dataset is a 1-week project; we want
  1-day projects.

## 5. The Soft Constraints (negotiable, but listen)

- **Backbone:** we are open to changing the backbone. ConvNeXt-Tiny
  is the current choice, but we have considered ResNet-50,
  EfficientNet-B0, MobileNet-V3-Large, and Swin-Tiny. We have
  not tested any of these.
- **Number of heads:** we are open to merging heads (e.g., body
  pose + head pose into a single 26-D output) or splitting them
  (e.g., PSR's 11 binary outputs into a single 11-D multi-label).
- **Loss design:** Kendall with clamp is the current balancer.
  We are open to GradNorm, PCGrad, or plain per-head weights.
- **Curriculum:** staged training (det → pose → all) is the
  current schedule. We are open to joint training from epoch 0
  with per-head LR ramps.
- **Auxiliary streams:** VideoMAE is the only auxiliary stream.
  We are open to dropping it or replacing it with something
  lighter.

## 6. The Target (what "success" looks like)

The user (Bashara) said: **"I want the model to really learn
something, not catastrophically fail. I am fine if I need to
change the backbone or anything."** The explicit target is
"80% confidence" on a 20-factor rubric, but the practical
target is:

**Within 7 days, we want a model that, on the 5% subset val
set, produces non-degenerate metrics on all 4 heads.** Not
state-of-the-art, not matching the paper, just non-degenerate:

- `det_mAP50` > 0.05 (currently 0.00; random would be ~0.001).
- `act_top5_accuracy` > 0.10 (currently 0.00; random over 74
  classes is ~0.067).
- `psr_overall_f1` > 0.05 (currently 0.00; class-imbalanced
  but a learnable head should be able to predict the most
  common pattern at > 5% F1).
- `position_MAE_mm` < 200 mm (currently 739; the model is at
  body scale, not mm scale, so < 200 is the right order of
  magnitude).

After we hit these targets, the next 30-day goal is to scale
to 10–20% subset and push metrics to half the paper's
baselines. The 6-month goal is to match or beat the paper
on the full IndustReal test set.

## 7. The Specific Questions (`04_HYPOTHESES_FOR_OPUS.md`)

The full list is in `04_HYPOTHESES_FOR_OPUS.md`. In short:

- **H1:** Kendall vs GradNorm vs PCGrad vs plain sum.
- **H2:** Staged curriculum vs joint training from epoch 0.
- **H3:** Drop the PSR causal transformer (sequence mode).
- **H4:** Drop VideoMAE (22M frozen params, no eval signal).
- **H5:** Class-balanced sampling + drop LDAM-DRW.
- **H6:** Smaller backbone (EfficientNet-B0) to free VRAM.
- **H7:** Drop or simplify FiLM conditioning on activity.
- **H8:** Replace the single-backbone design with 3 partial
  shared backbones (the most invasive option).

Each of H1–H8 is independently testable in 5 epochs at 5%
subset (~3.5h in FP32). We can run 2–3 of them per day.

## 8. The Deliverable (what we want from you)

**Five or more MD files, each a self-contained implementation
plan for one architectural change.** Each MD file must include:

1. **Problem statement** (1 paragraph). What's broken, with a
   citation to `02_COLLAPSE_CRISIS.md` or
   `evidence/metrics_baseline_pre_reinit.json`.
2. **Proposed change** (1–2 paragraphs). What to modify in
   `model.py`, `losses.py`, `train.py`, `config.py`, with
   line numbers.
3. **Code patches** (concrete, not pseudocode). Diffs or full
   replacement functions, ready to copy-paste.
4. **Hyperparameter changes** (table). What changes in
   `config.py`, with old → new values.
5. **Expected impact** (table). Which eval metrics should
   improve by how much, and which might regress.
6. **Experiment design** (1 paragraph). How to run the
   experiment, including the exact shell command.
7. **Success criteria** (1 paragraph). What "this worked"
   looks like in the eval output.
8. **Rollback procedure** (1 paragraph). How to revert if
   the change makes things worse.
9. **Risk assessment** (1 paragraph). What could go wrong,
   and how to detect it early (e.g., NaNDetector trip, loss
   spike, OOM).
10. **References** (1–5 papers or library docs, with URLs).

The MD files do not have to be long. 200–500 lines is the
right range. They do have to be **complete** — a code-reviewer
should be able to read the MD and produce a PR without
guessing.

**We prefer 5 focused MDs over 1 monolithic one.** Each MD
should be runnable on its own; you should not require
MD-A to be merged before MD-B can be tested.

## 9. The 7-Day Plan (how we will use your MDs)

If you give us 5 MDs, here is the schedule:

| Day | Action | MDs needed |
|---|---|---|
| 1 | Read all MDs, prioritize by impact/cost ratio. | All 5 |
| 2 | Run MD-1 (lowest-cost, highest-confidence change). | MD-1 |
| 3 | Re-eval. If pass, run MD-2. If fail, debug MD-1. | MD-1, MD-2 |
| 4 | Re-eval. Run MD-3. | MD-2, MD-3 |
| 5 | Re-eval. Run MD-4. | MD-3, MD-4 |
| 6 | Re-eval. Run MD-5. | MD-4, MD-5 |
| 7 | Final re-eval. Decide: continue iterating, or scale to 10% subset. | All 5 |

Each MD should be designed to take 1 day including re-eval.
If an MD takes longer, it is too big — split it.

## 10. What You Don't Need to Do

- **You don't need to write the master training script.** We
  have `run_reinit_fp32.sh` and `run_eval_post_retrain_fp32.sh`.
  Your MDs should reference these, not replace them.
- **You don't need to write the eval script.** We have
  `evaluate.py` and `eval_post_reinit.py`. Your MDs should
  reference these.
- **You don't need to design a new dataset.** We have
  `industreal_dataset.py`. Your MDs must work with the
  existing data format.
- **You don't need to optimize for the 1M-context model or
  the 200K-context model.** MiniMax-M3 is the default; use
  whatever context you need.

## 11. What We Will Provide (in this folder)

| File | Purpose |
|---|---|
| `00_JOURNEY.md` | 6-phase chronological narrative. |
| `01_WHAT_WE_BUILT.md` | Architecture tour. Read this before reading the code. |
| `02_COLLAPSE_CRISIS.md` | The 4-head collapse, in evidence-grade detail. |
| `03_CURRENT_RECOVERY.md` | The 4 fixes we just applied, with line numbers. |
| `04_HYPOTHESES_FOR_OPUS.md` | 7 specific architectural questions, with experiments. |
| `05_MASTER_PROMPT.md` | **This file.** |
| `code/config.py` | All hyperparameters. |
| `code/model.py` | The full architecture (2167 lines). |
| `code/train.py` | The training loop with reinit + FP32 (3733 lines). |
| `code/losses.py` | All losses including Kendall + LDAM + PSR-Focal (1505 lines). |
| `code/evaluate.py` | The 4-head eval (4004 lines). |
| `code/industreal_dataset.py` | The data loader (1383 lines). |
| `code/metrics.py` | Metric utilities (198 lines). |
| `code/eval_post_reinit.py` | The re-eval entry point. |
| `evidence/metrics_baseline_pre_reinit.json` | Eval output from the dead 39-epoch checkpoint. |
| `logs/train_digest.log` | Curated log excerpts (header, reinit, tail). |

## 12. The Closing Argument

We have a model that can produce non-zero features, run a
forward pass in < 500 ms, and train for 15 epochs without
collapsing. The architecture is **alive**. The training
procedure is killing it. Your job is to design 5 concrete
changes that, taken together, will let the model train to
non-degenerate metrics on all 4 heads within 7 days.

We are not asking for a research paper. We are asking for a
punch list.

The 39-epoch checkpoint is in
`/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth`.
The codebase is in
`/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/`.
The retrain is running as PID 2416305.

**Begin.**

— Bashara
