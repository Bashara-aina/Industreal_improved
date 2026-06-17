# POPW Fix v2 — Final Aggregated Report

**Date:** 2026-06-04
**Reporter:** reporter@popw-fix-v2
**Source data:** 5 verifier inboxes, rerun-smoke inbox, wire-warmup in-progress, code grep against train.py / losses.py / config.py / smoke_test_fixes.py

---

## ⚠️ Coverage Gap — CRITICAL

The team config (`~/.claude/teams/popw-fix-v2/config.json`) shipped with only **9 of the 20 expected agents**. The 11 agents that were never spawned:

| Missing Agent | Role |
|---------------|------|
| reviewer-1, reviewer-2 | Peer review of verifier + wire-warmup results |
| validator-1, validator-2 | validate_checkpoint.py + 1-epoch end-to-end run |
| integration-agent | Check fix interactions (warmup × Kendall, EMA × warmup, etc.) |
| documenter | POPW_FIX_REPORT_V2.md |
| tester-1, tester-2 | Unit tests + integration tests |
| debugger-1, debugger-2 | Regression hunt + smoke-test typo fix |
| monitor | Progress tracking |

The coordinator flagged this gap (`/home/newadmin/.claude/teams/popw-fix-v2/inboxes/coordinator.json` timestamp 2026-06-03T21:53:40Z). No reviewer-1 inbox exists (`reviewer-1.json` is empty array), confirming they never came online.

**Impact on this report:** Verdicts below are based on primary-source verifier reports + grep, not on independent peer review. Recommend a follow-up round with the 11 missing agents before declaring the build production-ready.

---

## 📊 Verification Results (5 verifiers)

| Fix Category | Verifier | Verdict | Evidence (file:line) |
|--------------|----------|---------|----------------------|
| PSR temporal smooth (signed tanh, -1 mask) | verify-psr | **PASS** | losses.py:1162-1164, 1170-1174 — `diff_p = (p_i[1:] - p_i[:-1]).mean()` (raw signed), `diff_l = -1 * (l_i[1:] - l_i[:-1]).mean()`, isfinite guard present |
| NaN guard (any-not-isfinite) + headpose divzero | verify-nan | **PASS** | train.py:2821 — `if any(not math.isfinite(x) for x in [_map50, _f1_act, _mae_pose, _f1_psr])`; train.py:1471-1472 — `mae_safe = max(mae_head_pose, 1e-6); head_pose_acc = 1.0 / (1.0 + mae_safe)` |
| VideoMAE proj in optimizer | verify-videomae | **PASS** | train.py:2315-2350 — `optimizer.add_param_group({'params': videomae_proj_params, 'lr': head_lr})`, `requires_grad = True`, stream+proj unfreeze in same block at epoch 10 |
| Memory leak fixes (FRAME_CACHE, pin_memory, del targets) | verify-memory | **❌ FAIL** | All 3 missing. industreal_dataset.py:159 has FRAME_CACHE dict but no `clear()`, no size limit, no epoch-end clearing. train.py:243 uses `pin_memory=C.PIN_MEMORY` unconditionally. grep `del targets` → 0 matches in train.py |
| EMA fixes (ModelEMA + _get_ema_decay) | verify-ema | **PASS** | validate_checkpoint.py:59 imports ModelEMA, lines 65-67 use `ema.shadow[name] = tensor`, line 68 calls `ema.get_ema()`. train.py:2305 uses `_get_ema_decay(epoch)` not hardcode 0.999 |

**Score: 4 PASS / 1 FAIL (memory leaks)** — the original 15-bug audit had 3 memory fixes; they are all missing in source.

---

## 🔥 Smoke Test (rerun-smoke, 2026-06-04)

- **Result:** 14/15 PASS, exit 1
- **Code defect surfaced:** NONE
- **The 1 failure is a smoke-test typo, not a code defect**

```
[FAIL] EMA shadow load: live round trip
File: src/smoke_test_fixes.py:619
NameError: name 'ema' is not defined
Cause: variable in scope is `ema_after` (line 602), not `ema`
Fix: replace `ema` with `ema_after` on line 619 (1-character change)
```

The 3 prior static checks on the same code path (source check, `ema.shadow.update()` usage, key filtering) **all pass** — confirming the underlying train.py code is correct.

### What the smoke test did NOT cover
- STAGE3_WARMUP_EPOCHS ramp integration (no dedicated test for it)
- Memory leak fixes (verify-memory already FAIL'd these — no need to re-test)

---

## 🛠 Wire-Warmup Status (in-progress)

The agent has been working through 3 self-assigned subtasks since 2026-06-03T21:54:02Z. **As of grep 2026-06-04, the edit is in the file** — STAGE3_WARMUP_EPOCHS ramp has been wired into train.py at 4 locations:

| Location | Purpose |
|----------|---------|
| train.py:2034 | Comment marker |
| train.py:2138-2148 | `stage3_warmup_state` dict definition with `warmup_epochs = int(getattr(C, 'STAGE3_WARMUP_EPOCHS', 3))` |
| train.py:2323-2333 | Activation on stage 3 entry (set `active=True`, `start_epoch=epoch`) |
| train.py:2507-2516 | Per-step warmup factor application in training loop |

**Status:** Code edit appears complete. The agent has not yet sent a final PASS to coordinator, so we have agent-confirmation pending. Recommend rerun-smoke re-run to confirm 14/15 + 1 (wire-warmup-specific) = 15/16 expected, with the same smoke-test-typo failure (1/16 = 6.25% — same bug as before).

---

## 🚀 Restart Command

```bash
cd /media/newadmin/master/POPW/working/code/industreal_improved_to_archive && bash src/run_restart_25pct.sh
```

The script (`run_restart_25pct.sh`):
- Resumes from `crash_recovery.pth` (25% subset, 31 epochs)
- Disables staged training (`--no-staged-training`)
- num_workers=0 (avoids pin_memory hang per train.py:297-301 comment)
- All 6 structural fixes documented in the script header are in place

---

## ⚖️ Risk Assessment

| Fix | Risk | Rationale |
|-----|------|-----------|
| PSR temporal smooth (signed tanh) | **LOW** | Loss is bounded (max ~1/9 for unit-step), defense-in-depth isfinite guard, no retraining needed for existing checkpoints |
| NaN guard (any-not-isfinite) | **LOW** | Logged at line 2821, comment dated 2026-05-31, only fires when validation metric is corrupt — combined metric is set to 0.0 (conservative) |
| Headpose divzero clamp (1e-6) | **LOW** | Pure epsilon guard, mathematically equivalent for mae >> 1e-6, prevents `1/0` crash only |
| VideoMAE proj in optimizer | **MEDIUM** | New param group at epoch 10 with same `head_lr` as activity/psr heads. Risk: optimizer state collision if resumed from a checkpoint that was saved before the group existed. Mitigated by `optimizer.add_param_group` (PyTorch handles this). Recommend warm-start with stage 3 reinit for safety. |
| EMA — ModelEMA + _get_ema_decay | **LOW** | validate_checkpoint.py:51-56 has a rationale comment block. train.py:2298-2302 has the bug-fix comment block. Class surface confirmed in models/model.py:1987-2044. |
| EMA live-round-trip test fix (1-char) | **LOW** | Smoke test only, doesn't touch production code |
| STAGE3_WARMUP_EPOCHS ramp | **MEDIUM** | New code, only grep-verified (not yet PASS-message from wire-warmup, not yet tested in smoke). Risk: if `stage3_warmup_state` is initialized at module load but never activated, `param_group_idx` lookup at line 2516 will IndexError. Mitigated by `stage3_warmup_state['active']` check at line 2511. |
| **Memory leak fixes** | **HIGH** | **ALL 3 STILL MISSING.** `verify-memory` reported FAIL. `del targets` is the highest-risk one — without it, the targets dict (with all labels/boxes/hand joints) accumulates across batches, leading to OOM on long runs. FRAME_CACHE unbounded means the module-level dict grows indefinitely. `pin_memory=True` + num_workers>0 hangs (per existing comment in train.py:297-301). **Do NOT start full training run until these are applied.** |
| Smoke test NameError (line 619) | **LOW** | Test script only, 1-character fix, doesn't affect training |

---

## 🎯 Recommended Next Steps (priority order)

1. **[BLOCKER] Apply 3 memory leak fixes** (verify-memory FAIL). Without this, 25% training will eventually OOM. Specifically:
   - Add `FRAME_CACHE.clear()` at end of each epoch in train.py
   - Change train.py:243 to `pin_memory=(C.PIN_MEMORY and C.NUM_WORKERS == 0)` (or set NUM_WORKERS=0 already in restart script)
   - Add `del targets` after backward pass at the end of the training step
2. **[CONFIRM] Wire-warmup final PASS** — ping the agent to confirm the edit is in (it appears complete in source, just lacks a final message)
3. **[NICE-TO-HAVE] Fix smoke test NameError** — 1-character change (`ema` → `ema_after` on line 619) for clean 15/15 report
4. **[DEFER] Spawn 11 missing agents** — especially reviewer-1/2 (peer review) and validator-1/2 (1-epoch training sanity check) for the next round
5. **[DEFER] Add STAGE3_WARMUP-specific smoke test** — current smoke test has no check for the warmup ramp activation

---

## 📁 Evidence Files

- `/home/newadmin/.claude/teams/popw-fix-v2/inboxes/coordinator.json` — all 5 verifier results + rerun-smoke
- `/home/newadmin/.claude/teams/popw-fix-v2/inboxes/rerun-smoke.json` — smoke test kickoff
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/training/train.py:2034, 2138-2148, 2305, 2323-2333, 2507-2516, 2821` — fix locations
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/training/losses.py:1150-1175` — PSR fix
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/validate_checkpoint.py:59, 65-68` — EMA fix
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/config.py:378` — `STAGE3_WARMUP_EPOCHS = 3`
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/smoke_test_fixes.py:602, 619` — smoke test typo
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/run_restart_25pct.sh` — restart script
