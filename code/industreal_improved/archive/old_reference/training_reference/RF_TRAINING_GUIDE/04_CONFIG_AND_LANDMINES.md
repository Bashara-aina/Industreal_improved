# Config & Landmines — What Resolves to What, and What Never to Touch

*The config in this project is layered. A value you "set" in `config.py` can be silently
overridden at runtime. This file maps that precedence so a setting never surprises you,
and lists the knobs that are load-bearing.*

---

## 1. Config precedence (LAST writer wins)

```
config.py module default
        ↓  overridden by
PRESET via apply_preset()        (e.g. PRESETS['stage_rf1'])
        ↓  overridden by
CLI ARG in train.py              (e.g. --detach-reg-fpn, --reinit-heads, --subset-ratio)
        ↓  overridden by
ENV VAR injected by stage_manager (e.g. _STAGE_LR_MULT, DET_GT_FRAME_FRACTION)
```

**Consequence:** what you read in `config.py` is *not* necessarily what ran. Always confirm
the **resolved** values from the log:

```bash
grep -E "DET_GT_FRAME_FRACTION|DETACH_REG_FPN|DETACH_PSR_FPN|MIXED_PRECISION|STAGED_TRAINING|BASE_LR" \
     src/runs/rf_stages/logs/train.log | head
grep -E "python.*train|--reinit-heads|--detach-reg-fpn|--detach-psr-fpn|--preset|--subset-ratio" \
     src/runs/rf_stages/logs/subprocess.log | head
```

---

## 2. ⚠️ THE landmine: `reinit_heads` silently turns on `--detach-reg-fpn`

In `stage_manager.py` (launch):

```python
if stage_cfg.get('reinit_heads'):
    cmd += ['--reinit-heads']
if stage_cfg.get('detach_reg_fpn', stage_cfg.get('reinit_heads')):   # ← defaults to reinit_heads!
    cmd += ['--detach-reg-fpn']
if stage_cfg.get('detach_psr_fpn', stage_cfg.get('reinit_heads')):
    cmd += ['--detach-psr-fpn']
```

So **any stage with `reinit_heads=True` and no explicit `detach_reg_fpn` key launches with
`--detach-reg-fpn`** → `train.py` sets `C.DETACH_REG_FPN=True` → regression gradient is
blocked from the backbone → `§Background-equilibrium`.

- `config.py` says `DETACH_REG_FPN = False`, but that default is **overridden** by the CLI
  flag this coupling injects. (This is exactly why RF1 ran detached while `paper_run` —
  launched manually, no reinit — did not.)
- **The fix** adds `'detach_reg_fpn': False` to the RF1 stage, breaking the coupling for RF1.
- **Still watch retries:** every entry in `RETRY_STRATEGIES` sets `reinit_heads=True`. So if
  **any** stage retries, the coupling re-injects `--detach-reg-fpn` for that retry unless
  that stage also has an explicit `detach_reg_fpn=False`. If you ever add reinit to RF2+,
  add the explicit `False` too.

**Always verify** (RF1 and any reinit retry):
```bash
grep -- "--detach-reg-fpn" src/runs/rf_stages/logs/subprocess.log   # must be EMPTY for RF1
```

---

## 3. Values `apply_preset()` derives for you (don't hand-set these wrong)

| Derived value | Rule | Resolves to |
|---------------|------|-------------|
| `DET_GT_FRAME_FRACTION` | det on, act+psr off | **0.9** (RF1, RF2) |
| | det on, act or psr on | **0.4** (RF3–RF10) |
| | det off | 0.0 |
| `TRAIN_DET/ACT/PSR/HEAD_POSE` | from preset `train_*` flags | per stage (file `01`) |
| `SUBSET_RATIO` | from `--subset-ratio` (stage cfg) | RF1 .20 … RF10 1.0 |

It logs the GT fraction every run (non-silent by design):
```
[config] DET_GT_FRAME_FRACTION = 0.90 (train_det=True, train_act=False, train_psr=False)
```
If that line disagrees with the stage you think you're running, you launched the wrong preset.

---

## 4. Env vars stage_manager injects (so manual runs differ from orchestrated runs)

| Env var | Meaning |
|---------|---------|
| `_STAGE_LR_MULT` | retry LR multiplier (1.0 / 0.2 / 0.1 / 0.05) |
| `_STAGE_WARMUP_MULT` | retry warmup multiplier |
| `_STAGE_SEED_OFFSET` | new seed per retry |
| `OUTPUT_ROOT_OVERRIDE` | where logs/ckpts go (`RF_RUN_DIR`) |
| `_STAGE_MANAGER_ACTIVE` | tells train.py it's orchestrated |
| `_STAGE_GATE_JSON` / `_STAGE_TARGET_MET_FILE` | gate spec + early-stop signal file |

**Implication:** a stage launched **manually** (no stage_manager) won't have these, so its
LR/warmup/seed/output path differ from the orchestrated launch. When reproducing a failure,
reproduce the **launch path** too.

---

## 5. DO NOT TOUCH (load-bearing — each was paid for in a prior collapse)

| Setting | Keep at | Why (the collapse it prevents) |
|---------|---------|--------------------------------|
| `mixed_precision` | **False** | AMP GradScaler silently skips steps → frozen model (RC-29) |
| `FOCAL_ALPHA` | **0.90** | α=0.75 collapsed at 173K:1 neg/pos; 0.90 gives positives net-positive gradient |
| `DET_ASYMMETRIC_GAMMA` / `DET_GAMMA_POS` | **True / 0.0** | positives get full (un-suppressed) gradient |
| `DET_OHEM_ENABLED` | **True** | unbounded negatives drive cls_mean to −16 |
| `USE_LDAM_DRW` | **False** (early joint stages) | s=30 → 1-class activity collapse |
| `KENDALL_LOG_VAR_*` clamps | as set | stop one head zeroing out another |
| `feature_bank_detach` | **True** | gradient through bank → double-backward crash |
| `use_geo_head_pose` | **True** | 6D→orthonormal rotation; strictly better than Euler |
| RF1 `detach_reg_fpn` | **False** | the whole reason RF1 now trains (§2) |

If you think one of these needs to change, change it **alone**, on a branch, and watch the
200-step board. Two changes at once = you learn nothing.

---

## 6. MAY tune (safe ranges, one at a time)

| Knob | Default | Safe range | When to move it |
|------|---------|-----------|-----------------|
| `BASE_LR` | 5e-4 | 3e-4 … 7e-4 | raise slightly if trunk learns but slowly; lower only for instability |
| `REINIT_REG_WARMUP_STEPS` | 1000 | 1000 … 2500 | raise if reg-shock (`§Reg-shock`) |
| `DET_EMPTY_SAMPLE` | 2048 | 1024 … 4096 | raise if det grad decays between GT batches |
| `DET_EMPTY_BG_SCALE` | 0.05 | 0.01 … 0.1 | raise to keep det alive on empty frames |
| `PSR_SEQ_EVERY_N_BATCHES` | 2 | 2 … 8 | **raise when `train_psr=False`** so det trains every step |
| `PRIOR_PROB`/`pi` | 0.01 | 0.01 … 0.05 | secondary; denser early signal, more early FPs |
| stage `gate` thresholds | per file 01 | — | lower to achieved+margin if `§Gate-too-strict` |
| `batch_size`/`grad_accum` | 4/8 | keep product = 32 | OOM (`§OOM`) — 2/16 is proven |

---

## 7. EMA checkpoints — evaluate the right weights

- RF stages run `use_ema=True`. With `STAGED_TRAINING=False` (all RF presets), EMA tracks
  from epoch 0.
- **The checkpoint you report/resume must be the EMA / `best` weights**, not a raw or
  init-blended snapshot. A `best.pth` that is an EMA lagging near init will poison the next
  stage's resume. Confirm `best` holds real trained weights before advancing
  (sanity: its val metrics should match the epoch it was saved at).

---

## 8. The "is my run configured correctly?" one-shot check

Run this right after launch; every line should match the stage you intend:

```bash
L=src/runs/rf_stages/logs
echo "== resolved config ==" ; grep -E "DET_GT_FRAME_FRACTION|MIXED_PRECISION=|STAGED_TRAINING=|DETACH_REG_FPN" $L/train.log | head
echo "== launch flags ==" ;    grep -E "train\.py|--preset|--reinit-heads|--detach-reg-fpn|--subset-ratio" $L/subprocess.log | head
echo "== step-0 det ==" ;      grep "DET-INIT" $L/train.log | head
echo "== first liveness ==" ;  grep -E "LIVENESS_GRAD|backbone:" $L/train.log | head
```

Expected for **RF1** after the fix:
- `DET_GT_FRAME_FRACTION = 0.90`, `MIXED_PRECISION=False`
- launch shows `--reinit-heads` **and NOT** `--detach-reg-fpn`
- `DET-INIT cls_mean ≈ -4.6`
- first `backbone:` line says **ALIVE**

If those four are right, RF1 is configured to succeed. Hand off to `00 §4` and run the loop.
