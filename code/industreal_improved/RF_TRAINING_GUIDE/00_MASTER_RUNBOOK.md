# RF1 → RF10 Master Runbook

*Your operational guide to a smooth progressive-training journey. Read this file first.*

---

## 0. What "flawless" actually means (read this once, then never worry again)

No training run is literally error-free, and any guide that promises that is lying to
you — that promise is exactly what produced 29 consult files and a stuck RF1. **A smooth
journey is not the absence of problems. It is catching every problem in the first ~200
steps instead of after an 8-hour GPU run, and having a pre-decided action for each one.**

This guide makes your journey smooth by making every known failure **boring and
pre-solved**:

- You will never again watch a run for hours that was already dead at step 500 — because
  you check the **kill-or-keep signals at 90 seconds, 200 steps, and end of epoch 1**.
- You will never again guess "lower the LR?" — because the **failure tree** (file `03`)
  maps each symptom to one action.
- You will never again be surprised by a gate — because you know the **realistic target
  for each stage** (file `01`) before you launch.

Confidence comes from the checklist, not from hope. Run the checklist and you advance.

---

## 1. The five files

| File | Use it for |
|------|-----------|
| `00_MASTER_RUNBOOK.md` (this) | The universal loop you run for every stage; golden rules; orchestration model |
| `01_STAGE_PLAYBOOK.md` | Per-stage cards RF1–RF10: targets, what changes, the #1 risk of each stage |
| `02_HEALTH_SIGNALS.md` | Decode every log line; healthy-vs-sick reference numbers |
| `03_FAILURE_TREE.md` | Symptom → cause → exact command. The "it broke, what now" file |
| `04_CONFIG_AND_LANDMINES.md` | Config precedence, runtime-override traps, do-not-touch list |

---

## 2. The Prime Directive

> **No run is allowed past 200 optimizer steps until the 200-step health board is green.**

90% of the pain in this project came from violating this one rule. A bad run costs you
nothing if you kill it at minute 2. It costs you a day if you let it finish.

---

## 3. One-time setup (before you start the ladder at all)

Do this **once**, on the training box, before RF1.

```bash
# 1. Get the fix that makes RF1 trainable (regression gradient reaches the backbone,
#    head_pose on, trunk grad-norm probe). Already committed on this branch.
git pull origin claude/youthful-fermi-xxdcwl

# 2. Confirm the environment is FP32 + single-GPU sane
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Confirm GPU is empty (RF1 retry #0 died of OOM because other procs held VRAM)
nvidia-smi   # expect ~0 MB used by other processes

# 4. Confirm the dataset / GT coverage once (so you never blame the model for missing labels)
python diag_gt_coverage.py --preset stage_rf1 --subset-ratio 0.20   # if this script exists in your tree
```

**Non-negotiable invariants** (verify in `04_CONFIG_AND_LANDMINES.md`):
- `mixed_precision: False` in every preset (AMP silently skips steps → frozen model).
- `DETACH_REG_FPN` resolves to **False** for RF1 at runtime (the fix). Verify with the
  launch-line check in §5 below.
- `BASE_LR = 5e-4`, effective batch = 32 (`batch_size 4 × grad_accum 8`).

---

## 4. The universal per-stage loop

Run this identical loop for **every** stage RF1…RF10. The only thing that changes between
stages is the **target numbers** (file `01`) and **which heads must be ALIVE** (file `02`).

```
┌─ PRE-FLIGHT ──────────────────────────────────────────────────────────┐
│ • Open the stage card in 01_STAGE_PLAYBOOK.md → note gate targets       │
│ • Confirm resume checkpoint exists (RF1: latest; RF2+: previous best)   │
│ • Confirm the launch command's CLI flags (see §5)                       │
└────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─ LAUNCH ──────────────────────────────────────────────────────────────┐
│ python3 -m src.training.stage_manager --start                          │
│ (or launch the single stage manually — see §6)                         │
└────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─ 90-SECOND CHECK (process is alive & not OOM) ────────────────────────┐
│ • PID alive? GPU mem stable (not climbing toward 12 GB)?                │
│ • Step-0 line printed: [DET-INIT] cls_mean ≈ -4.6 (pi=0.01)?           │
│ • No traceback in subprocess.log?                                       │
│   FAIL → 03_FAILURE_TREE.md §OOM or §Crash                             │
└────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─ 200-STEP HEALTH BOARD (the decisive gate) ──────────────────────────┐
│ All of these must be true (full reference in 02_HEALTH_SIGNALS.md):     │
│ • RC-29: committed > 0, skipped = 0                                     │
│ • LIVENESS: every head that should train is ALIVE (>1e-6)              │
│ • LIVENESS: backbone ALIVE (NOT STARVED) ← the new probe              │
│ • DET_HEALTH: cls_mean MOVING off -4.7 (toward -3 .. -2)              │
│ • No NaN/inf in any loss                                                │
│   ANY FAIL → kill now, go to 03_FAILURE_TREE.md. Do not "let it cook." │
└────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─ END OF EPOCH 1 (direction check) ────────────────────────────────────┐
│ • DET_PROBE verdict moving away from pure "LOCALIZING": score_p50 > │
│   0.02 and rising; preds>0.30 becoming non-zero                        │
│ • Validation produced a NUMBER for the gate metric (det_mAP50 ≠ NaN). │
│   If the gate metric is NaN, the gate can never pass → 03 §Gate-NaN    │
└────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─ STEADY MONITOR (every epoch) ────────────────────────────────────────┐
│ • Gate metric improving toward target (file 01)                        │
│ • No head went DEAD; no loss spike > spike factor                      │
│ • Kendall log_var values stay in [-4, 2]                              │
└────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─ GATE & ADVANCE ──────────────────────────────────────────────────────┐
│ • All gate metrics ≥ target (MAE ≤ target) → stage_manager advances    │
│ • Save/confirm best.pth → it becomes the resume source for next stage  │
└────────────────────────────────────────────────────────────────────────┘
```

**The rule of thumb for time budget:** if the 200-step board is green, the stage will
almost certainly reach its gate. If it's red, no amount of waiting fixes it. Your job is
the first 200 steps; the rest is monitoring.

---

## 5. Verify the launch command (catch the silent-override trap)

The single most damaging bug in this project's history was a CLI flag the launcher
injects **silently**. Always confirm what the stage actually launched with:

```bash
# What flags did stage_manager pass to train.py?
grep -E "python.*train|--preset|--reinit-heads|--detach-reg-fpn|--detach-psr-fpn" \
     src/runs/rf_stages/logs/subprocess.log | head

# What did the config RESOLVE to at runtime?
grep -E "DET_GT_FRAME_FRACTION|DETACH_REG_FPN|MIXED_PRECISION|STAGED_TRAINING" \
     src/runs/rf_stages/logs/train.log | head
```

For **RF1**, after the fix, you must see `--reinit-heads` **but NOT** `--detach-reg-fpn`
(reg gradient must reach the backbone). For RF2+ you should also not see
`--detach-reg-fpn` unless a retry re-injected it (see file `04` §runtime traps).

---

## 6. How the orchestrator decides (so its behavior never surprises you)

`stage_manager.py` runs a check cycle. Per stage it computes four checklists, then
`decide_action()` picks one move:

| Checklist | What it tests | Source |
|-----------|---------------|--------|
| **gate** | All target metrics met? (`≥` thresh, or `≤` for MAE) | `evaluate_gate()` |
| **health** | Heads ALIVE, no consecutive-DEAD streak | `evaluate_health()` |
| **convergence** | Metric improving faster than `min_improvement`/window | per-stage cfg |
| **stability** | No crashes/OOM, no loss-spike streak | per-stage cfg |

`decide_action()` logic (verbatim from code):

```
stability FAIL   → kill_and_retry
health   FAIL    → kill_and_retry
convergence FAIL → kill_and_retry
gate PASS        → advance_stage
otherwise        → continue
```

Retries escalate through `RETRY_STRATEGIES`: default → reduce_lr_5x → reduce_lr_2x_warmup_2x
→ reduce_lr_10x_warmup_2x → reduce_lr_20x_warmup_3x, then **escalate to human** once
`retry_count ≥ RETRY_ESCALATION_THRESHOLD`.

### ⚠️ The doom-loop you must know about

**Every retry reduces the learning rate AND re-injects `--reinit-heads`.** For a
*starved-trunk* failure (RF1's classic mode), lower LR makes escape **harder**, and each
reinit throws away progress. So:

- A genuine starvation/equilibrium failure must be fixed by **changing the recipe**
  (file `03`), **not** by letting the auto-retry cut the LR five times.
- If you see the gate metric returning **NaN**, the gate is `UNKNOWN`→`FAIL` forever and
  the orchestrator will retry-with-lower-LR until it escalates. That is a **measurement**
  bug (eval not producing mAP), not a model bug. Go to `03 §Gate-NaN`.

**Operator override:** when the 200-step board is green but a *transient* made the gate
fail, relaunch the stage at **base LR** (default strategy), not the reduced-LR retry.

---

## 7. Golden rules (violating any of these is how every prior run died)

1. **FP32 always.** Never flip `mixed_precision` to True. (RC-29: AMP GradScaler silently
   skips `optimizer.step()` → 4 identical eval cycles → "frozen model".)
2. **Watch the trunk, not just the heads.** A head can be ALIVE while the backbone is
   STARVED. The new `backbone:ALIVE/STARVED` probe is your truth signal for "is the model
   actually learning features?"
3. **Never reduce LR to fix a starvation/equilibrium problem.** Reduced LR is for
   *instability* (loss spikes, NaN), not for "won't fire."
4. **One change per relaunch.** If you change two knobs and it works, you've learned
   nothing and you can't reproduce it. Change one thing, observe, decide.
5. **A gate metric of NaN is never a pass and never the model's fault first** — check the
   evaluator before touching the model.
6. **Don't advance on a soft gate.** The gates exist so RF10 inherits a healthy model.
   Skipping a gate just moves the failure downstream where it's harder to diagnose.
7. **Checkpoints are the contract between stages.** RF1 resumes `latest`; RF2–RF10 resume
   the previous stage's `best`. If `best.pth` is an init-blend (EMA lag), you poison the
   next stage — confirm `best` holds real trained weights.

---

## 8. The "I just want the commands" quickstart

```bash
# Start / resume the whole ladder (orchestrated):
python3 -m src.training.stage_manager --start

# Check current state without touching anything:
python3 -m src.training.stage_manager   # runs cmd_check()

# Tail the signals that matter:
tail -f src/runs/rf_stages/logs/train.log | grep -E "DET_HEALTH|LIVENESS|DET_PROBE|RC-29|backbone:"

# Kill a dead run (then fix per file 03, then relaunch):
kill <PID>
```

Now open `01_STAGE_PLAYBOOK.md` and find the card for the stage you're about to run.
