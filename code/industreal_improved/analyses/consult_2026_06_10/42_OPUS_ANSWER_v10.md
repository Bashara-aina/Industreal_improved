# 42 — OPUS ANSWER v10: detach_reg_fpn Is the Smoking Gun (2026-06-21)

> Response to `41_OPUS_MASTER_PROMPT_v10.md`. Opus read the actual code, not just the summary.

---

## TL;DR

**The plateau is substantially dynamic (a config regression), not structural.** RF2's backbone is being denied dense box-regression gradient because `detach_reg_fpn` resolves to **True** for `stage_rf2`. The RF1 fix (which got 0.184) was applied only to RF1's stage config — `stage_rf2` preset silently re-enabled it. LR restarts do nothing because you can't anneal your way out of a severed gradient path.

---

## Verifiable Facts from Code

| Claim | Code Says | Verdict |
|-------|-----------|---------|
| `DETACH_REG_FPN` for RF2 | `config.py:1117` — `stage_rf2` has `'detach_reg_fpn': True`. rf2 `stage_cfg` has **no** override key. CLI never fires. | **CONFIRMED True** |
| Top-k IoU floor exists? | `losses.py:139,152` — floor exists. `config.py:307` — `DET_POS_IOU_IOU_FLOOR=0.2` (committed). | **Already coded** — needs restart |
| Per-class AP persisted? | `evaluate.py:267,273` — computed, **thrown away**. | **NOT persisted** |
| Combined metric formula | `train.py:2153-2196` — for RF2: `0.667·mAP50 + 0.333·(1/(1+MAE))` | **Misleading — 1/3 is head-pose** |

---

## Unified Diagnosis

The backbone is shaped by classification + head_pose + conditioning **only** — box regression is detached (`model.py:561`, `reg_feat = feat.detach()`). Your own RF1 fix comment (`stage_manager.py:108-121`) describes the consequence verbatim:

> *"...that leaves the backbone with ONLY the sparse classification path — features never become object-discriminative, so the cls conv cannot separate fg from bg and the head sticks at the ... 'predict-background-everywhere' equilibrium (localizes but won't fire)."*

This is a one-to-one match for symptoms: **bestIoU 0.86–0.98 (localizes) + mAP 0.20 + 12/24 classes AP=0 (won't fire).** It also reconciles all contradictory evidence:

- **POS_ANCHOR_PROBE 0.64–0.80**: On the easy 12-16 classes, cls-shaped features *are* good enough. The probe samples img 0/1 — never shows dead classes.
- **Pseudo-classing +50%**: Gap IS the dead classes — feature-starvation produces class-selectively (discriminable classes survive; subtle/small/rare ones collapse).
- **LR restart = zero effect**: Rules IN this diagnosis. A detached gradient path is not a local minimum.

**One caveat**: RF2 (detach=True) reaches 0.20, slightly **above** RF1 (detach=False) at 0.184. So detach is a **handicap that v8 fixes + 2.5× data partly compensated for** — flipping it should move the ceiling but may not single-handedly clear 0.40. Pair with per-class AP logging.

---

## Recommendations

### Don't advance to RF3 — fix RF2 first
- RF3 `stage_rf3` preset **also** has `detach_reg_fpn: True` — advancing inherits the bug
- Gate isn't close (0.20 vs 0.40 gate)
- Combined-metric "progress" is artifact
- Fix is nearly free: one config line + IoU floor + restart

### Tier 0 — alongside live run (15 min, zero risk)
1. **Persist per-class detection AP** (already computed at `evaluate.py:267` — log it)
2. **Run `scripts/overfit_50img_cls.py`** (separate process, doesn't disturb live run)

### Tier 1 — the corrected restart
3. Set `stage_rf2` `detach_reg_fpn=False` in `config.py:1117`
4. Add belt-and-suspenders: `'detach_reg_fpn': False` to rf2 `stage_cfg`
5. Keep `DET_POS_IOU_IOU_FLOOR=0.2`
6. Restart from current `best.pth`, keep trained heads, don't reinit
7. Watch 3–4 epochs with per-class AP logging

### Decision rule after Tier 1
- Dead classes wake + mAP climbs past ~0.25–0.30 in 3-4 epochs → detach was the bottleneck
- Overfit hit 0.8+ but run stays flat → data-scale/assignment on dead classes
- Overfit can't reach 0.8 on 50 images → target/assignment/eval pipeline bug

### The one missing measurement
**Per-class AP + per-class positive count + per-class max-anchor-IoU.** Each leaves a different signature:
| If you see... | It's... |
|---|---|
| Dead classes have **high positive counts** but AP=0 | feature starvation (→ detach fix) |
| Dead classes have **~0 positives** + max-IoU < 0.3 | anchor mismatch |
| Dead classes get positives only at IoU 0.2–0.3 | top-k poisoning (floor=0.2 should help) |
| Dead classes have positives, good IoU, but wrong class | label noise / confusion pair |

---

*Saved from Opus consultation round 10. Code-grounded diagnosis identifying `detach_reg_fpn=True` as the primary cause of the 6-epoch plateau.*
