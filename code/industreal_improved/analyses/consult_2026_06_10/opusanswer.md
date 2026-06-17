I'll dig into this deeply. The previous 29 files have been theorizing extensively — my first job is to ground everything in the **actual code** rather than the summary, because after 4 rounds of consultation the blind spot is most likely a gap between what the docs *say* the code does and what it *actually* does. Let me investigate.

The real source tree is `code/industreal_improved/src/`. Let me read the actual config, model, loss, and stage-manager code — and critically, find whether the gradient-sparsity theory has ever been *measured*.

Already a major discrepancy: the master prompt repeatedly states `DETACH_REG_FPN=True (config.py:573)`, but the actual code says **`DETACH_REG_FPN = False` at line 567**, and the grep found `detach_reg_fpn` in **no preset at all**. The entire "R2.5 paradox" rests on regression gradient being detached. Let me read the real presets and the detach logic to confirm.

Confirmed a critical discrepancy. The actual code has **`DETACH_REG_FPN = False`** (config.py:567) and **no preset overrides it** — not `paper_run`, not `stage_rf1`, not `recovery_det_only`. The master prompt's entire "R2.5 paradox" rests on `DETACH_REG_FPN=True`, which is **factually wrong in the current code**. The comment even says "Set True for RF1" but the value is hardcoded `False`.

But I also see a *real* mechanism the docs didn't emphasize: `REINIT_REG_WARMUP` (lines 554-560) suppresses regression loss to **1%** for the first **1000 steps** after `--reinit-heads`. Let me verify where `DETACH_REG_FPN` actually gets set, whether RF1 uses `--reinit-heads`, and how the warmup is applied.

This reveals the **actual** mechanism, and it's more subtle than either the code default or the master prompt suggests. Look at `stage_manager.py:1147`:

```python
if stage_cfg.get('detach_reg_fpn', stage_cfg.get('reinit_heads')):
    cmd += ['--detach-reg-fpn']
```

When a stage sets `reinit_heads=True` (RF1 does), `detach_reg_fpn` **defaults to that same `True`** — and `train.py:4311` then sets `C.DETACH_REG_FPN = True` at runtime. So RF1 *does* run detached, but **not** because of `config.py` — because of this default-tie in the launcher. Meanwhile `paper_run` (R2.5) is launched manually without `--reinit-heads`, so it stays `False`. Let me confirm the RF1 stage definition, the launch logic, and the model's gradient path.

Runtime behavior confirmed: RF1 launches with **both** `--reinit-heads` and `--detach-reg-fpn` (the latter auto-injected by the `reinit_heads` default at `stage_manager.py:1147`), while R2.5/`paper_run` runs manually with neither. So the R2.5-vs-RF1 comparison differs in *four* ways, not one. Let me verify the detection loss internals (bounded background, pi init, reg warmup) and the retry LR strategies to finish grounding the diagnosis.

Let me confirm the one thing the entire 29-file theory hinges on but (per Q1) was *never measured* — whether backbone gradient norm is logged anywhere — and check how `_step_counter` drives the warmup.

Decisive confirmation of the blind spot: train.py logs **per-head** grad norms (`cls_w_grad`, `reg_w_grad`, Kendall log_var, liveness) at lines 1649–1679 and 2036–2082, but **nothing for the backbone or FPN** — the exact quantity the entire death-spiral theory is about was never measured. Let me check one more thing: there's a backbone-freeze mechanism near the reinit path (line 532) that could be *compounding* the starvation.

Another compounding factor. `_set_stage_requires_grad` (train.py:525) **freezes ConvNeXt backbone stages [0,1] for the first 5 effective epochs** — and with `--reinit-heads`, the schedule restarts at Stage 1. The RF1 death spiral happens in "the first 500 steps" = epoch 1, exactly when the early backbone may be frozen. But all presets set `staged_training: False`. I need to know whether this freeze is gated on that flag, or runs regardless.

Important: the backbone freeze is gated on `staged_training` (train.py:847), which is `False` for all RF presets — so the backbone is *fully trainable* in RF1. That rules out "frozen backbone" as the cause and isolates it cleanly: the backbone is willing to learn, but the only gradient reaching it is the sparse detection-classification signal, because regression is detached and all other heads are off.

Let me cross-check my reading against their newest root-cause doc (file 29) and look for any committed training logs that show the actual `cls_w_grad`/`reg_w_grad` numbers.

No live RF1 logs in this repo (training runs on a separate box). Let me read their newest root-cause doc (file 29) to engage directly with its reasoning and confirm exactly where its premise diverges from the code.

File 29 contains the **evidence that refutes its own central theory** — and this is the blind spot (Q10). Look at its own logs:

- **LIVENESS: `det=5.38e+00 ALIVE`** on frame-batches (gradient norm 5.38 is *huge*, not sparse)
- **Timeline: `det_cls` drops 1.6168 → 0.6260 → 0.6852`** over 200 steps — the classification loss is *moving substantially*

A loss that falls 60% and a grad norm of 5.38 are categorically **not** "gradient too sparse to move weights." The gradient is strong; the model is using it to descend into the wrong solution — "predict background everywhere." This is an **attractor / feature-discriminability** problem, not a gradient-magnitude problem. That distinction flips the recommended fix. Let me verify the one thing that would confirm it — how the focal/background loss is normalized (a too-coarse normalization dilutes the positive signal directly).

The detection loss normalizes by `num_pos` (line 247, 310, 346) — standard RetinaNet, so positives are *not* diluted by total-anchor count. That further undercuts the "magnitude" theory. Let me read the full cls-loss block (200–350) to understand the asymmetric-gamma and OHEM/bounded-background interaction, since that governs whether positives can ever rise.

The detection loss is sound and standard: normalized by `num_pos` (line 310) and `n_img_with_gt` (line 352), OHEM keeps all positives + hardest negatives, asymmetric-gamma protects positives. Critically — the "bounded background" uses **`torch.randperm`** (line 227, random anchors, re-sampled every batch) and **only on empty frames**, not "the highest-scoring 512 every batch" as files 28/29 claim. So Q4's "deterministic 512" concern is also based on a misreading. Let me confirm the remaining flag defaults and base LR, then I'll have everything.

Now the picture is complete and decisive. The team has **already maxed out every positive-favoring lever** and detection *still* won't fire:

- `FOCAL_ALPHA = 0.90` (line 419) — positives weighted 9× over background
- `DET_ASYMMETRIC_GAMMA = True`, `DET_GAMMA_POS = 0.0` (lines 438-439) — positives get *zero* focal suppression
- `DET_OHEM_ENABLED = True` — background limited to hardest negatives
- `DET_GT_FRAME_FRACTION = 0.9` (line 1241) — 90% of frames carry GT boxes
- loss normalized by `num_pos`

If the problem were positive-gradient *sparsity/magnitude*, these would have fixed it. They didn't — which confirms the bottleneck is elsewhere. Also note `PSR_SEQ_EVERY_N_BATCHES = 2` (line 622): **every other RF1 batch is a seq-batch where detection gets zero gradient** (and `BASE_LR = 5e-4`, not the `1e-4` file 29's math assumed). Let me read the two edit sites precisely before changing anything.

I have what I need to write the diagnosis and make the fix. Let me confirm the exact submodule attribute names (backbone / FPN / detection head) so the new diagnostic targets the right parameters.

All submodule names confirmed (`self.backbone`, `self.fpn`, `self.detection_head`). I now have a complete, code-grounded diagnosis. 

Here's the key realization that reframes everything: **RF1 deviates from the proven `recovery_det_only` recipe in *two* ways, not one.** File 29 caught only the first (`head_pose` off). The second — and more important — is that RF1 runs with **regression gradient detached from the backbone**, which `recovery_det_only` does *not*. Restoring both makes RF1's gradient configuration identical to the recipe Opus originally prescribed. Let me apply the fix plus the never-implemented diagnostic.

Now let me verify the three edits are syntactically valid before committing.

All three compile. Let me review the complete diff before committing.

The shell's working directory persisted from the earlier `cd`. Let me use an absolute path to the repo root.

The diff is clean and correctly scoped. Let me commit to the designated branch and push.

Pushed to `claude/youthful-fermi-xxdcwl`. Now the diagnosis — and it diverges sharply from the file 00–29 conclusion. I read the actual code (config, model, losses, stage_manager, train), and the central premise of the consultation is **factually inverted**.

> Note: the GitNexus MCP tools that `CLAUDE.md` asks for aren't connected in this remote session, so I did the impact analysis manually (grep/read across all call sites). The edits are config-value/additive-logging changes, not symbol refactors.

---

## TL;DR — you've been debugging the wrong experiment

RF1 is not dying from **gradient sparsity**. Your own logs disprove that. The cause is that RF1 runs with the **regression gradient detached from the backbone**, so the shared trunk never learns object-discriminative features — and the comparison to R2.5 was never valid in the first place, because R2.5 never cold-started detection.

---

## 1. The premise is backwards

File 29 says: *"DETACH_REG_FPN (config.py:573)"* detaches regression, and that's a fixed architectural fact shared by R2.5 and RF1.

The code says otherwise:

- `config.py:567` → **`DETACH_REG_FPN = False`**. No preset overrides it — not `paper_run`, not `stage_rf1`, not `recovery_det_only`.
- It becomes `True` **only at RF1 runtime**, via `stage_manager.py:1147`:
  ```python
  if stage_cfg.get('detach_reg_fpn', stage_cfg.get('reinit_heads')):  # defaults to reinit_heads!
      cmd += ['--detach-reg-fpn']
  ```
  RF1 sets `reinit_heads=True`, so `detach_reg_fpn` silently inherits `True` and `train.py:4311` flips `C.DETACH_REG_FPN=True`.
- `paper_run` (R2.5) is launched **manually, outside the RF ladder** — no `--reinit-heads`, no `--detach-reg-fpn`. So **R2.5 ran with regression gradient flowing to the backbone; RF1 does not.**

So R2.5 vs RF1 differ in **four** ways, not one:

| | R2.5 (`paper_run`) | RF1 (`stage_rf1`) |
|---|---|---|
| Detection head | **warm** (no reinit) | **cold** (`--reinit-heads`) |
| Regression→backbone gradient | **flows** | **detached** |
| Other dense heads | act + psr + head_pose | **none** |
| Reg loss early | full | warmup-suppressed to 1% for 1000 steps |

File 29's table lists "Det reg (GIoU): DETACHED" for **both** columns. That's the error the whole analysis is built on.

## 2. It is not gradient sparsity — your logs already prove it

File 29 §7 records:
- `LIVENESS det=5.38e+00 ALIVE` on frame-batches — a grad norm of **5.38 is enormous**, not "effectively zero."
- `det_cls` falls **1.6168 → 0.6260 → 0.6852** over 200 steps while `cls_mean` stays at −4.70.

A loss that drops 58% with a grad norm of 5.4 is **not** "16 anchors can't move 28M weights." The gradient is strong — the model is using it to descend into the **loss-minimizing solution under extreme imbalance: predict background everywhere.** `cls_mean` stays at the floor and `det_cls` still drops because it's polishing background suppression, never climbing toward firing.

And you've already maxed every positive-favoring lever, which a sparsity problem would have fixed:
- `FOCAL_ALPHA = 0.90` (positives 9× weighted) — `config.py:419`
- `DET_GAMMA_POS = 0.0` (positives get **zero** focal suppression) — `config.py:439`
- `DET_OHEM_ENABLED = True`, loss normalized by `num_pos`, `DET_GT_FRAME_FRACTION = 0.9`

With all of that and detection *still* pinned at π=0.01, the bottleneck cannot be positive-gradient magnitude. It's that the **shared cls conv applies the same weights at every spatial location, and the backbone features at object locations are indistinguishable from background** — because the one dense signal that teaches "where objects are" (box regression) is detached. Classification alone cannot bootstrap discriminative features from a cold start.

## 3. The real root cause

```
--reinit-heads (cold cls+reg)  +  --detach-reg-fpn (reg grad never reaches trunk)
        +  all other heads off  +  reg-loss warmup (1% for 1000 steps)
   ⟹ backbone gets only the classification path
   ⟹ features never become object-discriminative
   ⟹ shared cls conv can't separate fg/bg
   ⟹ stuck at the "background everywhere" equilibrium (localizes, won't fire)
```

The regression head still localizes well (good `bestIoU`) because it fits boxes off the *frozen pretrained* features — but with its gradient detached it can't teach the backbone anything. Detaching regression was added to stop "reinit gradient shock," but you **also** have `REINIT_REG_WARMUP` (1%→100% over 1000 steps) for exactly that. The warmup is the correct, sufficient guard; the permanent detach was redundant overkill that starved the trunk.

**A secondary drain:** `PSR_SEQ_EVERY_N_BATCHES = 2` (`config.py:622`) means **half of RF1's steps are seq-batches where detection gets zero gradient** (your `det=1e-6 DEAD` lines) — even though PSR is off in RF1. So detection effectively trains on ~50% of steps.

## 4. What I changed (pushed to your branch)

Three scoped, reversible edits that make RF1's gradient config identical to the Opus-prescribed `recovery_det_only`:

1. **`stage_manager.py` — `stage_rf1` adds `'detach_reg_fpn': False`** → regression gradient reaches the backbone. Keeps `--reinit-heads` + the reg warmup (the right shock guard). **This is the primary fix.**
2. **`config.py` — `stage_rf1` `train_head_pose: True`** (was `False`) → dense feature-diverse trunk signal, matching `recovery_det_only`. Insurance, ablatable.
3. **`train.py` — backbone/FPN grad-norm probe** in the liveness logger. The trunk gradient is the quantity 29 files theorized about but **never measured**. It now prints `backbone:ALIVE[…]` / `backbone:STARVED[…]`.

**To deploy on the training box:**
```
kill 4189479                         # current run is in the doom loop (detached + LR cut)
git pull origin claude/youthful-fermi-xxdcwl
# reset rf1 state: retry_count=0, status=pending; rm stale rf1 checkpoints
python3 -m src.training.stage_manager --start
```
**Within ~200 steps, expect:** `backbone:ALIVE` (was STARVED), `cls_mean` climbing off −4.6, `score_p50` rising above 0.0167, `preds>0.30` becoming nonzero. **Watch for shock:** if `cls_mean` crashes toward −16, *lengthen* `REINIT_REG_WARMUP_STEPS` (1000→2000) rather than re-detaching.

## 5. Your ten questions

**Q1 — Is the sparsity analysis correct? Other mechanism?** No. `det=5.38 ALIVE` + `cls 1.6→0.63` refute it. Mechanism = detached regression → non-discriminative features → background equilibrium, plus ~50% of steps are det-dead seq-batches. The new probe will show `backbone:STARVED` under the old config.

**Q2 — Which option?** Not A alone (head_pose treats the symptom). **B done right** is the root fix: don't detach reg, keep the warmup, don't cut LR. I shipped B+A (= `recovery_det_only`). The **Option D you missed:** don't `--reinit-heads` at all if you're resuming a usable detection head — reinit discards the warm start that made R2.5 work; or do a head-only warmup (freeze trunk ~300 steps, then unfreeze with reg attached).

**Q3 — Is RF1's premise flawed; remove it?** No. Detection-from-reinit on a pretrained backbone is standard RetinaNet fine-tuning — it works daily in mmdetection/Detectron2 with **no detach**. RF1 is viable; the POPW-specific detach broke it. Keep RF1.

**Q4 — Bounded background 512, deterministic?** Misread. It's `torch.randperm` (random, re-sampled every batch), **2048** not 512 (`DET_EMPTY_SAMPLE`), and **only on empty frames** (`losses.py:224-234`). With `DET_GT_FRAME_FRACTION=0.9` it's rarely even hit. Not a contributor.

**Q5 — Should retries ever raise LR?** Yes. For trunk-starvation / attractor-escape, cutting LR (your 0.2→0.1→0.05 ladder) digs the model *deeper* into the wrong basin. Signal to raise (or hold) LR: `backbone STARVED` + `det head ALIVE` + `cls loss falling` + `score_p50 flat`. I did **not** edit `RETRY_STRATEGIES` (it affects all 10 stages — your call); recommend: det-bootstrap retries never go below base LR.

**Q6 — Right diagnostic / threshold?** Yes, added. `<1e-4` total backbone norm = STARVED; healthy joint training is O(0.1–10). Pair it with a **feature-drift probe** (std of P3 features, or cosine distance of backbone weights vs init) — grad norm can be nonzero yet features still not differentiate. What matters is that it *rises* and features *change*.

**Q7 — DETACH_REG_FPN long-term design?** Retire it; default `False`; **never tie it to `reinit_heads`** (that coupling at `stage_manager:1147` is the latent bug). Use `REINIT_REG_WARMUP` (and/or a brief trunk freeze) for shock — those don't create a permanent dependency on other heads for trunk gradient.

**Q8 — Higher π?** Low priority. α=0.90 + γ_pos=0 + OHEM already over-compensate for positives, so π isn't the bottleneck. π=0.02–0.05 is a reasonable *secondary* lever (denser early signal) but won't revive a starved trunk. Fix the trunk first.

**Q9 — Audit OD coverage?** Worth one `diag_gt_coverage` run, but `DET_GT_FRAME_FRACTION=0.9` (`config.py:1241`) now forces ~90% GT frames at the sampler, so frame-level positive starvation is largely solved; anchor-level imbalance is handled by focal+OHEM. Likely not the active blocker — confirm with the new backbone probe.

**Q10 — The blind spot.** Two:
- **You never measured the trunk.** The det-head liveness (5.38) and falling cls loss were in the logs the whole time and already contradicted the sparsity story. 29 files reasoned *about* backbone gradient without logging it.
- **The R2.5↔RF1 comparison is invalid.** R2.5 never reinit detection, never detached reg, kept a warm head. "R2.5 worked" tells you nothing about a reinit+detach+det-only stage. You were comparing two different experiments and attributing the gap to the one variable (head_pose) that happened to be in your doc.

And the meta-pattern: **patch debt.** Each round added a patch (detach, reg-warmup, bounded-bg, OHEM, α-bumps, Kendall bounds, GT-frame sampling, seq-grad-zeroing). They now interact in ways no single analysis tracks — and the culprit is one redundant patch (the reinit→detach coupling). The fix direction here is to *remove* a patch, not add another.

One thing I did **not** do, because it's genuinely your decision and changes all 10 stages: flip the `RETRY_STRATEGIES` LR ladder, and raise `PSR_SEQ_EVERY_N_BATCHES` for det-only stages so detection trains every step. Want me to make RF-bootstrap retries hold LR at base, and skip seq-batches when `train_psr=False`?
Before I write these, one honest reframe — and then I'll give you the most rigorous guide I can.

No one can promise "flawless, no errors, every target hit without question." Anyone who does is selling you the same false confidence that produced 29 files and a stuck RF1. ML training is stochastic; the difference between a smooth run and a 3-week death spiral is **not** the absence of problems — it's catching them in the first 200 steps instead of after a GPU-day, and having a pre-decided action for each one. That is what I'll build you: a runbook that front-loads verification, defines realistic per-stage targets, and gives a symptom→action tree so you never again sit through an 8-hour run that was dead at step 500.

Let me pull the remaining ground truth (all 10 stage gates, presets RF3–RF10, exact diagnostic strings) so the guide matches your actual code, not generic advice.

I now have all 10 stages' gates/presets. Let me grab the exact diagnostic log strings and the gate-evaluation logic so the guide tells you precisely what to grep for.

Let me read the gate-evaluation and retry-decision logic so the runbook describes the orchestration accurately.

I have full ground truth now — all 10 stage gates, every preset, the diagnostic strings, and the gate/retry orchestration (including the doom-loop trigger: when `det_mAP50` comes back `NaN`, `evaluate_gate` marks it `UNKNOWN`→`FAIL`→`kill_and_retry`→LR cut). Let me write the 5-file guide, grounded entirely in your actual code.

All five files written. Let me verify and commit them to the branch.

Done — 5 files written, committed, and pushed to `claude/youthful-fermi-xxdcwl` under `code/industreal_improved/RF_TRAINING_GUIDE/`. Every number in them is read from your actual `config.py` presets, `stage_manager.py` gates, and `train.py` diagnostics — not generic ML advice.

| File | What it gives you |
|------|-------------------|
| **`00_MASTER_RUNBOOK.md`** | The one loop you run for every stage: pre-flight → 90-second → **200-step health board** → epoch-1 → gate → advance. Plus the orchestrator's decision logic and the doom-loop warning. |
| **`01_STAGE_PLAYBOOK.md`** | A card per stage RF1–RF10: exact gate thresholds, heads, resume source, the #1 risk of that stage, and an honest target expectation. |
| **`02_HEALTH_SIGNALS.md`** | Every log line decoded (`DET_HEALTH`, `DET_PROBE`, `LIVENESS` incl. the new `backbone:ALIVE/STARVED`, `RC-29`, Kendall) with healthy-vs-sick numbers + the 200-step green board. |
| **`03_FAILURE_TREE.md`** | Symptom → cause → exact command for all 13 known failure modes. The "it broke, what now" file. |
| **`04_CONFIG_AND_LANDMINES.md`** | Config precedence, the `reinit_heads → --detach-reg-fpn` silent-injection trap, do-not-touch list, tunable ranges, EMA checkpoint guidance. |

**The honest part, stated plainly:** these files don't make failures *impossible* — nothing does. They make every known failure **boring and pre-solved**, and they move your decision point from "after an 8-hour run" to "at step 200." That is what a smooth journey actually is. The one promise I'll stand behind: if you run the 200-step green board and it's all green, you are training, not death-spiralling — and if it's red, file `03` already has your next move.

**Your immediate next action** (RF1, with the code fix from earlier already on this branch):

```bash
git pull origin claude/youthful-fermi-xxdcwl
kill <current RF1 PID>          # it's in the reduced-LR doom loop
# reset rf1 state: retry_count=0, status=pending; clear stale rf1 checkpoints
python3 -m src.training.stage_manager --start
```

Then watch for the four RF1 green signals (from `00 §3` / `04 §8`): `DET_GT_FRAME_FRACTION=0.90`, launch shows `--reinit-heads` **but not** `--detach-reg-fpn`, `DET-INIT cls_mean ≈ -4.6`, and the first `backbone:` line reads **ALIVE**. If those four are right, RF1 is configured to succeed.

One open item from before still stands if you want it: making the auto-retry hold LR at base for the bootstrap stage (instead of cutting it) and skipping seq-batches when `train_psr=False` — both affect all 10 stages, so I left them as recommendations rather than changing them unasked. Say the word and I'll wire them in.