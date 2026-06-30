# 69 — Opus Response to the Final-Push Consult (files 64–68)

Date: 2026-06-30 · Model: Opus 4.8 · Method: read 64–68 in full, verified every
load-bearing claim against `origin/main` source. Findings cite code. Where I could not
verify (no data/GPU here), I say so.

> Note: files 64–68 live on `main`; this response is committed to the working branch and
> will sit alongside them once merged.

---

## TL;DR — two corrections that change your plan, then the 10 decisions

1. **File 65 is solving a non-problem.** Head-pose training already optimizes
   *normalized* direction. `head_pose_loss_split` (`losses.py:941–965`, called at
   `losses.py:1515`) L2-normalizes both prediction and target forward/up vectors
   *before* the MSE (lines 954–958), and the docstring says so: *"Direction term uses
   L2-normalized vectors so it is scale-invariant."* The un-normalized `pose.csv`
   magnitude (0.014–0.030) is therefore irrelevant to what the model learns. **8.71° is
   already what you train and already what you eval. Do NOT normalize GT at load time —
   it's a redundant no-op, and the feared "2500× MSE jump" (file 65 Q3) cannot happen
   because the direction term divides the scale out regardless.** Spend that time
   elsewhere. (Real caveat below: the *up* vector, not the fix.)

2. **Subprocess eval: do it, Option A, everywhere, reading the checkpoint you already
   save.** Concrete design in §Decision-5. This is your highest-value infra change and
   the idle RTX 3060 has a real job here.

Everything else is judgment; answers below.

---

## The 10 decisions (file 68)

### Decision 1 — Is the simple head the right architecture?
**Yes, keep it as-is.** `LayerNorm→Linear(512→256)→GELU→Dropout→Linear(256→75)`
(`model.py:1377`) is correct: short gradient path, ~150K params, fed the joint
projection that already contains `det_conf ⊕ GAP(c5_mod_blend) ⊕ GAP(p4)`
(`model.py` activity_proj). Do **not** add the suggested variants now:
- (a) deeper MLP → more overfitting capacity on 3.7k frames. No.
- (b) residual connections → pointless across a 2-layer MLP. No.
- (c) separate det_conf path → det_conf is *already* in the projection input;
  re-injecting it is redundant. No.
One change worth making only **if collapse persists**: raise dropout to 0.3 and add
weight decay on the head (overfitting, not capacity, is the risk on this data). Ship
the current head first; don't pre-optimize.

### Decision 2 — Go/no-go metric for "simple head works"
Use **prediction diversity**, not top-1, as the first-epoch gate. Top-1 can look fine
while collapsed (predict the majority class → top-1 ≈ majority prevalence). Compute on
the val set after epoch 1:
- **#distinct predicted classes ≥ 15** (out of ~26 classes with ≥1% support), AND
- **mean prediction entropy ≥ ~1.5 nats** (collapse → near-0), AND
- **act_macro_f1 > 0.01**.

If all three hold → the head escaped the collapse attractor; continue. If
#distinct ≤ 3 → it collapsed again; then (and only then) switch activity loss from
CE+label-smoothing to **class-balanced focal** and/or raise head dropout. Do **not**
gate on `act_top1 > 0.05` in epoch 1 — too strict, too noisy on 1.9k val frames.

(Cheap instrumentation: in the activity eval, log `np.bincount(preds)` and
`-(p*log p).sum(1).mean()`. ~5 lines.)

### Decision 3 — Raise DET_GT_FRAME_FRACTION 0.4 → 0.6 for RF4?
**No, keep 0.4.** Detection is your *most* likely task to pass; activity is the
*at-risk* one. Raising the GT-frame fraction concentrates batches on detection-bearing
frames and reduces the activity/PSR class diversity per batch — it would help your safe
task at the expense of your fragile one. Only raise it if detection visibly stalls
(<0.15 mAP50 by epoch ~8). Priorities, not micro-gains.

### Decision 4 — OneCycleLR pct_start for the simple head?
**Leave pct_start=0.1.** A 150K-param MLP with Xavier init converges in a handful of
epochs; it does not need a 30% warmup. Longer warmup mainly protects large
from-scratch modules — not this. Changing it is noise. (If you change anything on LR,
the higher-leverage knob is a slightly higher head LR via the existing param group,
but with `ACTIVITY_LR_MULTIPLIER=1.0` and 5e-4 you're already fine.)

### Decision 5 — Subprocess eval design (A / B / C?)
**Option A, for *all* validation calls** (gate + full). Reasons: the 200-batch gate
eval is exactly where hangs occur (file 64 §"Key Constraints" #4), so exempting it
defeats the purpose; and maintaining two code paths (C) is not worth it. Overhead is
~5–10 s import + load per eval × ~50 evals ≈ 8 minutes total against a 70-hour run —
negligible.

Concrete, minimal design (≈80 lines):

```python
import multiprocessing as mp, json, os
_CTX = mp.get_context('spawn')          # NOT fork: fork shares the parent CUDA context

def _val_worker(ckpt_path, out_path, overrides):
    import os, sys, json, torch
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # run eval on the IDLE RTX 3060
    sys.path.insert(0, 'src')
    from src import config as C
    for k, v in overrides.items(): setattr(C, k, v)
    from src.models.model import POPWMultiTaskModel
    from src.evaluation.evaluate import evaluate_all
    model = POPWMultiTaskModel(...).to('cuda').eval()
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda')['model'])
    loader = build_val_loader(...)            # NUM_WORKERS=0
    with torch.no_grad():
        metrics = evaluate_all(model, ..., max_batches=overrides.get('MAX_BATCHES'))
    json.dump(metrics, open(out_path, 'w'))

def run_val_subprocess(ckpt_path, out_path, overrides, timeout=900):
    p = _CTX.Process(target=_val_worker, args=(str(ckpt_path), str(out_path), overrides))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.kill(); p.join()                    # SIGKILL — frees the child CUDA context
        return {}
    return json.loads(open(out_path).read()) if os.path.exists(out_path) else {}
```

Key points (these answer file 64 Q1–Q7):
- **No separate serialization.** Read `latest.pth` — the pre-val checkpoint you already
  write (`train.py:4343`). Skips file 64's 15 s `_val_state.pkl` step entirely.
- **`spawn`, not `forkserver`.** `forkserver` + CUDA on Python 3.13 is fragile; the
  5–8 s spawn import is fine at this cadence. (file 64 Q2)
- **`CUDA_VISIBLE_DEVICES=0` in the child** puts eval on the idle 3060, fully isolating
  it from the training context on GPU 1. A killed eval can never corrupt training. This
  is the right use of GPU 0 — far better than batch-splitting (Decision 7). (file 64 Q3, file 68 Q7)
- **After `p.kill()` the OS reclaims the child's entire CUDA context** — separate
  process, separate context. The parent needs no `empty_cache` for the child's memory,
  and killing mid-kernel cannot corrupt the parent. That isolation is the whole point.
- **Two-tier timeout (file 64 Q7):** v1 — single 900 s timeout. If you want the
  refinement, have the worker `touch` a heartbeat file after model-load and after every
  20 batches; parent kills if heartbeat is >120 s stale. Add it after v1 works.

### Decision 6 — Auto-load crash_recovery.pth?
**Resume-only, never promote.** Auto-load `crash_recovery.pth` to *continue training*
when its mtime is newer than `latest.pth` — that's just resuming optimization, safe.
**Never** copy it into `best.pth` (those weights are post-train/pre-val, unvalidated;
promoting them can regress your tracked best). Concretely: in the resume logic
(`train.py:3608–3618`) pick the newest of {latest, crash_recovery} for the *training
state*, but leave `best.pth` selection driven only by validated metrics. (file 64 Q5)

### Decision 7 — Use GPU 0 via naive batch split (not DDP)?
**No.** Manual cross-device batch splitting needs the model replicated on both GPUs
with manual gradient reduction — you'd be reimplementing `DataParallel`, which is
*slower* than single-GPU for a 6 GB model once you pay PCIe transfer + sync to the
slower 3060. Net negative. **Use GPU 0 for the eval subprocess instead** (Decision 5) —
that's the genuinely useful split and it costs zero training throughput.

### Decision 8 — Does the file-66 abstract work for AAIML?
**The pivot to AAIML is sound** (ML/AI venue, IEEE Xplore, live Oct-10 deadline, content
fit). But the abstract has three honesty/risk problems — fix before submission:
1. **"five tasks" + "head pose within 1° of SOTA"** overclaims. PSR and activity may not
   work, and head pose is *forward direction only* — your **up vector is at ~95° MAE
   (file 65), i.e. essentially unlearned.** Reframe to *"head orientation (gaze/forward
   direction)"* and state plainly which of the five tasks succeed and which you analyze
   as failures. Reviewers punish "5 tasks" when 2 are at zero far harder than they
   punish an honest "3 succeed, 2 we dissect."
2. **Claim 3 ("probe misreading wasted 10 days", rated ★★★★★) should NOT be a headline
   contribution.** As a standalone it reads as a debugging anecdote and invites
   "you should have known." Keep it as a *methods caution* inside §5, generalized
   ("per-parameter liveness probes are routinely misread as head-level gradient
   magnitudes; here is a worked example and a correct measurement"). Lead instead with
   the **temporal-head/sampler mismatch** — that *is* a generalizable, genuinely
   interesting finding (a balanced per-frame sampler silently destroys a temporal head's
   input). That's your strongest novel result.
3. **Verify the efficiency numbers (46M params, 85 GFLOPs, 4.8 FPS) before they go in.**
   With the simple head the param count changed; recount with `thop`/`fvcore`.

Suggested honest headline: *"A multi-task assembly-verification model on a single
consumer GPU: which tasks share a backbone gracefully, which don't, and why."*

### Decision 9 — Head-pose disclosure?
Given correction #1, the **8.71° forward MAE is honest and needs no special disclosure**
— it's both trained and evaluated on normalized direction. What you **must** disclose:
report it as *forward/gaze direction*, not full head pose, and **report the up-vector
MAE (~95°) or drop the up component**. Hiding a 95° up error while claiming "head pose
at SOTA" is the actual reviewer risk — not the magnitude artifact you were worried about.

### Decision 10 — Run a single-task baseline?
**Yes — but the SAME backbone, detection-only, not YOLOv8m.** A `ConvNeXt-T + det head`
trained for a few epochs is the *controlled* ablation that makes "multi-task trade-offs"
a measured claim instead of an assertion; YOLOv8m is a different architecture/training
recipe and muddies the comparison. ~2 h of GPU time for your paper's central thesis is
the best-spent compute in the whole project. Do it.

---

## Answers to the sharper sub-questions in 64/65/67

- **64 Q4 / 67 Q8 (validate every N / full-eval cost):** with subprocess eval the hang
  risk collapses, so keep per-epoch *gate* eval (200 batches) and run *full* eval every
  2–3 epochs and once at each stage end. A 38k-frame full eval at 1.2 batch/s is ~9 h —
  schedule it only at RF10 final, not per epoch.
- **65 up-vector (Q5):** ~95° means the up direction is unlearned, not a normalization
  bug (the loss already normalizes it). Likely the GT up vectors are noisy/under-constrained
  for an egocentric head. Either drop the up term or report it honestly; don't expect
  load-time normalization to move it.
- **65 position units (Q2/Q4):** `HEAD_POSE_POS_SCALE` standardizes position to O(1) for
  the `pos_loss` term (`losses.py:947–953`); the eval reports mm. Verify the eval's
  ×1000 matches the CSV's true unit on one recording by hand — a 10-minute check — before
  trusting `position_MAE_mm=118`.
- **67 Q3 (drop PSR?):** if PSR macro-F1 is still 0 after the simple-head run, **scope to
  4 tasks and analyze PSR as a documented failure** (transition objective needs true
  sequence batches; per-frame focal on 95%-static labels is near-constant-optimal — you
  already diagnosed this). A clean 4-task system + 1 dissected failure is a stronger
  paper than 5 half-working tasks.
- **67 Q6 (when to switch to sequence_mode):** only switch the activity head back to the
  temporal stack (`ACTIVITY_HEAD_SIMPLE=False`) if/when you train on real
  `sequence_mode` batches. On per-frame data the simple head should match or beat the
  temporal head — that head-to-head *is* your §5.1 ablation. Don't switch mid-RF.
- **67 Q9 (per-stage checkpoints):** **yes, save `best_rfN.pth` per stage.** Cheap
  insurance and you need them for the ablation table. Overwriting one `best.pth` across
  stages is a mistake.
- **67 Q10 (seed variance):** for the paper, run the *final* config with 3 seeds on the
  cheapest representative stage and report mean±std on the headline metrics. Reviewers
  expect variance; single-seed numbers invite "is this cherry-picked."

---

## What I did NOT change in code this turn
Files 64–68 are planning/consult docs and the relevant code fixes (simple head, bank
bypass, RAM cache, LR reset) are already on `main` and look correct
(`model.py:1374–1419`, `:2193–2199`). The two concrete code tasks that remain —
subprocess eval (Decision 5) and the diversity/entropy go-no-go instrumentation
(Decision 2) — I can implement on request; I held off because they touch the live
training/eval loop and you have a run in progress (PID 3618126). Say the word and I'll
write both.

## The one thing to watch
Everything downstream hinges on the simple head's **epoch-1 prediction diversity**
(Decision 2). If it stays diverse, your RF schedule (file 67) is reasonable and the
paper (4 honest tasks + the sampler/temporal-mismatch analysis) is real. If it collapses
to ≤3 classes again, the binding constraint is data (46/72 classes <1%), not
architecture — and the paper becomes "why frame-level 72-way AR is infeasible at this
scale," which is *also* publishable, just a different §5. Either branch is a paper;
neither is the "match SOTA on 5 tasks" framing. Hold that line.
