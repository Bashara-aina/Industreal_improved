# STAGE_MANAGER_VERIFICATION

**Date:** 2026-07-14
**Target:** `src/training/stage_manager.py` + `src/config.py` + `src/training/train.py`
**Purpose:** Verify PAPER_OUTLINE §3.4 "3-stage curriculum" claim against live code

---

## 1. Claim vs Reality: The "3-Stage" Discrepancy

| Claim (PAPER_OUTLINE §3.4) | Reality (stage_manager.py) | Verdict |
|----------------------------|---------------------------|---------|
| "3-stage RF1-RF3 progressive unlocking" | 10 stages (RF1-RF10) defined at lines 82-437 | **MISMATCH** — paper understates by 7 stages |
| "3-stage" implicitly suggests completion at RF3 | RF4-RF10 continue with progressive data scaling and tightening gates | **MISMATCH** — paper omits 70% of the curriculum |

**Impact on paper:** §3.4 must either (a) describe RF1-RF10, or (b) describe RF1-RF3 as "initial unlocking phase" and RF4-RF10 as "consolidation/scaling phase." The current framing is materially incomplete.

---

## 2. Complete RF1-RF10 Stage Definitions

All stages defined at `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/stage_manager.py`, lines 82-437.

| Stage | Active Heads | Subset Ratio | Max Epochs | Gate Thresholds |
|-------|-------------|-------------|------------|-----------------|
| **RF1** | det only | 20% | 20 | `det_mAP50_pc >= 0.15` |
| **RF2** | det + pose | 50% | 30 | `det_mAP50_pc >= 0.22` AND `forward_angular_MAE_deg <= 70.0` |
| **RF3** | det + pose + act | 35% | 15 | `det_mAP50_pc >= 0.20` AND `act_top1 >= 0.05` AND `forward_angular_MAE_deg <= 70.0` |
| **RF4** | all + PSR | 50% | 20 | `det_mAP50_pc >= 0.20`, `act_top1 >= 0.06`, `psr_f1_at_t >= 0.05`, `pose <= 65.0` |
| **RF5** | consolidate | 50% | 10 | `det_mAP50_pc >= 0.22`, `act_top1 >= 0.08`, `psr_f1_at_t >= 0.06`, `pose <= 60.0` |
| **RF6** | scale 65% | 65% | 10 | `det_mAP50_pc >= 0.24`, `act_top1 >= 0.10`, `psr_f1_at_t >= 0.08`, `pose <= 55.0` |
| **RF7** | continue 65% | 65% | 10 | `det_mAP50_pc >= 0.24`, `act_top1 >= 0.12`, `psr_f1_at_t >= 0.10`, `pose <= 50.0` |
| **RF8** | scale 80% | 80% | 10 | `det_mAP50_pc >= 0.26`, `act_top1 >= 0.14`, `psr_f1_at_t >= 0.12`, `pose <= 45.0` |
| **RF9** | scale 90% | 90% | 10 | `det_mAP50_pc >= 0.28`, `act_top1 >= 0.16`, `psr_f1_at_t >= 0.14`, `pose <= 40.0` |
| **RF10** | final 100% | 100% | 15 | `det_mAP50_pc >= 0.30`, `act_top1 >= 0.18`, `psr_f1_at_t >= 0.16`, `pose <= 35.0` |

### Head Unlocking Progression

```
RF1:  [det]                                    (1 head)
RF2:  [det + pose]                              (2 heads)
RF3:  [det + pose + act]                        (3 heads)
RF4+: [det + pose + act + psr]                  (4 heads, complete)
```

### Data Scaling Progression

```
RF1:  20% → RF2: 50% → RF3: 35% → RF4: 50% → RF5: 50%
→ RF6: 65% → RF7: 65% → RF8: 80% → RF9: 90% → RF10: 100%
```

Note the non-monotonic drop at RF3 (50%→35%) — possibly to compensate for adding the activity head.

---

## 3. Gate Threshold Triggers (evaluate_gate)

**Function:** `evaluate_gate()` at stage_manager.py line 1693

The gate function uses key aliasing to map metric file keys to gate thresholds:

| Gate Field | Metric Key Alias | Notes |
|-----------|-----------------|-------|
| `det_mAP50_pc` | direct lookup | Used across all 10 stages |
| `act_top1` | also tried as `act_clip` | Aliased key fallback |
| `psr_f1_at_t` | direct lookup | Appears from RF4 onward |
| `forward_angular_MAE_deg` | direct lookup | Also shortened to `pose` in RF4+ gates |

**Behavior:**
- Returns `(True, metrics_dict)` when ALL threshold conditions are met
- Returns `(False, failed_metrics_dict)` when any threshold is not met, with per-task pass/fail details
- Missing metric keys are handled gracefully (logged but not fatal)

---

## 4. Near-Gate Advancement

**Function:** `_check_near_gate_advance()` at stage_manager.py line 1341

When a stage does not fully pass the gate, near-gate advancement allows proceeding if close enough:

| Stage Range | Near-Gate Threshold | Also Requires |
|-------------|--------------------|---------------|
| RF1-RF3 | >= 85% of gate met | Stalled (no >1% improvement over last 3 epochs) |
| RF4-RF7 | >= 90% of gate met | Stalled |
| RF8-RF10 | Strict (no near-gate) | N/A |

A "stalled" check evaluates metric history — if no metric has improved >1% over the last 3 epochs, the stage is considered converged and allowed near-gate advance.

**Log signals:** `[NEAR-GATE]` prefix on near-gate advancement decisions.

---

## 5. Decision Engine (decide_action)

**Function:** `decide_action()` at stage_manager.py line 2343

Priority order (first match wins):

1. **Health check**: If retry_count >= max_retries (3), return 'escalate'
2. **Stability check**: If training crashed (no PID / crashed), return 'kill_and_retry'
3. **Convergence check**: If epoch >= dynamic max_epochs, return 'advance_stage' (force advance — gate becomes advisory)
4. **Gate check**: If gate passed, return 'advance_stage'
5. **Near-gate check**: If near-gate threshold met and stalled, return 'near_gate_advance'
6. **Default**: return 'continue'

**Dynamic max_epochs** (`_compute_dynamic_max_epochs()` at line 1411):
- If still improving at 80% of max_epochs: extends by 50%
- If plateaued for `patience_epochs`: cuts short
- This means RF1 (nominally 20 ep) could run up to 30 ep if still improving

---

## 6. LR Schedule Verification (vs PAPER claims)

| Paper Claim | Verified Config | Verdict |
|------------|----------------|---------|
| §3.4 implies a cosine schedule | `ONE_CYCLE_LR=True`, `USE_COSINE_ANNEALING=False` | **MISMATCH** — uses OneCycleLR, not cosine |
| "SWA + LR decay" for RF3 stabilization | `USE_SWA=False` in config.py | **MISMATCH** — SWA is disabled |
| LR schedule unspecified | OneCycleLR with `pct_start=0.1`, `steps_per_epoch=1`, 2-epoch warmup (LinearLR) | **Needs explicit description in paper** |

### Confirmed Scheduler Architecture (train.py lines 4230-4316)

```
Warmup (LinearLR, start_factor=1e-5, end_factor=1.0, total_iters=2)
  → OneCycleLR (max_lr from config, pct_start=0.1, steps_per_epoch=1)
  → [no SWA]
```

- `ONE_CYCLE_PEAK_FACTOR="auto"` resolves to `effective_batch / 32` (Goyal et al. linear scaling rule)
- Effective batch = 6 (batch_size) × 8 (grad_accum) = 48, so peak factor = 48/32 = 1.5
- Checkpoint restore re-applies max_lr from config (F4b fix, train.py lines 4407-4489)

---

## 7. Legacy Epoch Scheme vs Stage Manager Epochs

The config.py contains a **separate, legacy** curriculum scheme that is NOT used by stage_manager:

| Parameter | Legacy Value | Notes |
|-----------|-------------|-------|
| `STAGED_TRAINING` | `False` | Legacy flag — NOT active |
| `STAGE1_EPOCHS` | 5 | Legacy — DIFFERS from stage_manager RF1=20 |
| `STAGE2_EPOCHS` | 10 | Legacy — DIFFERS from stage_manager RF2=30 |
| `STAGE3_EPOCHS` | 85 | Legacy — DIFFERS from stage_manager RF3=15 |

**Risk:** A reader who searches only config.py for stage/epoch values would find these legacy numbers and report incorrect curriculum epochs. The stage_manager values override these entirely when the manager is active.

---

## 8. Debug Log Line Verification

The `cmd_check()` function (stage_manager.py line 2432) produces these identifiable log lines during curriculum execution:

| Log Signal | Function | Line |
|-----------|----------|------|
| `"Stage Manager Check @ {timestamp}"` | `cmd_check()` entry | ~2435 |
| `"Current stage: {name} [{status}]"` | Stage status report | ~2440 |
| `"Advancing to {name}: {description}"` | Stage transition | ~2470 |
| `"Gate PASSED for {name}!"` | Gate success | ~2480 |
| `"[NEAR-GATE] ..."` | Near-gate advancement | ~1350 |
| `"Training PID {pid} alive, epoch {epoch}"` | Health check | ~2450 |
| `"Killing PID {pid}"` | Training termination | ~2490 |
| `"Launching {name}: preset={preset}, epochs={max_epochs}"` | New stage launch | ~1670 |
| `"estimate_progress()= {pct}%"` | Progress estimate | ~1300 |

These log lines enable runtime verification that the curriculum is executing as designed.

---

## 9. Recommended Paper Revisions for §3.4

Based on all findings, the paper's §3.4 should be revised to state:

1. **10-stage curriculum** (not 3-stage): RF1-RF10 with progressive head unlocking, data scaling, and tightening metric gates
2. **LR schedule**: OneCycleLR (not cosine) with 2-epoch warmup, dynamic peak factor per linear scaling rule; SWA disabled
3. **Gate-based advancement**: metric thresholds must be met per stage; near-gate fallback at 85-90% for early stages
4. **Dynamic epochs**: automatic 50% extension if still improving at 80% of nominal max

---

## 10. Verification Summary

| # | Check | Result |
|---|-------|--------|
| 1 | RF1-RF3 triggers exist (stage_manager.py:82-437) | PASS |
| 2 | Gate thresholds documented (evaluate_gate, line 1693) | PASS |
| 3 | 3-stage claim matches reality | **FAIL** — 10 stages, not 3 |
| 4 | Near-gate advancement at 85% for RF1-RF3 | PASS |
| 5 | OneCycleLR confirmed (not cosine, SWA off) | PASS |
| 6 | Debug log lines confirmed (cmd_check, line 2432) | PASS |
| 7 | Legacy epoch scheme vs stage_manager discrepancy | **WARN** — STAGE1_EPOCHS=5 vs RF1=20 |

**Overall:** The curriculum implementation is robust and well-engineered, but the paper's §3.4 claim of "3-stage" curriculum is materially incomplete and must be revised before submission.
