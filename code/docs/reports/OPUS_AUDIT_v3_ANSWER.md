# Opus Audit v3 — Answer (2026-06-11)

Response to `MASTER_PROMPT_v3.md`. Every claim below was verified by reading
the source in this repository (commit `77b2615`), not the summary. **All
fixes described here are APPLIED in this branch** — this is an audit + patch
set, not just a report.

---

## Overall verdict (Q8)

**As uploaded, the implementation was NOT safe to run: the recovery
configuration was a complete no-op, and three of the safety guards could
never fire.** Twelve issues were found; the seven blockers are now fixed in
this branch. After these fixes, the recovery run is safe to launch — with the
caveats in §5.

The single worst finding: **`--preset recovery` did nothing.** `apply_preset`
was invoked on a different module object than the one the model, losses, and
training loop read (config split-brain), so `zero_det_conf`, FP32, and
`staged_training=False` were all silently ignored — and because
`MIXED_PRECISION` had been flipped back to `True` and `USE_MIXUP`/`CUTMIX`
were still on, the "recovery" run would have trained in **fp16 (the confirmed
R2 NaN mode), with cutmix label corruption (RC-15), det_conf un-zeroed, and
EMA-blended best.pth** — four of the previously diagnosed killers
simultaneously.

---

## 1. Blockers found and fixed

| # | Issue | Where (pre-fix) | Fix applied |
|---|-------|-----------------|-------------|
| B1 | **Config split-brain / preset no-op.** `train.py` applied presets via `import config as _cfg_mod` (root module object) while train/model/losses read `from src import config`. The root `config.py`/`model.py` were *symlinks* — identical code, but Python still creates two separate module objects with independent state, so `apply_preset('recovery')` mutated globals nobody read. `evaluate.py:61` also reads the root object. | `src/training/train.py:3582`, root `config.py`, root `model.py`, `src/evaluation/evaluate.py:61` | Root `config.py` and `model.py` replaced with alias modules (`sys.modules[__name__] = src.<module>`) so **all import paths resolve to one object**; `train.py` now calls `C.apply_preset(...)` directly and logs the resulting flags; unknown preset names now crash instead of warning. Verified at runtime: `import config is src.config → True`, preset flags visible through both names. |
| B2 | **`MIXED_PRECISION = True` regression.** fp16 AMP is the confirmed R2 failure mode (first NaN at `backbone.0.conv1.weight`; `diag_amp_nan.py`/`diag_amp_2step.py`). | `src/config.py:290` | Default `False` with the evidence cited in a comment. Re-enable only after a dedicated AMP smoke run. |
| B3 | **CutMix logit-mixing still active (RC-15).** `USE_MIXUP=True`, `CUTMIX_ALPHA=1.0`; `cutmix_activity` mixes the *output logits* (train.py:467), builds `images_mixed` (461-462) and never feeds it to the model; LDAM argmaxes the soft label → wrong-label supervision; gate `epoch%2==1, stage>=3` fires during any resumed/odd-epoch run. | `src/config.py:298,452`; `src/training/train.py:404,461-476,1085-1090` | `USE_MIXUP=False`, `CUTMIX_ALPHA=0.0` with full rationale; `'use_mixup': False` added to the recovery preset. The implementation is left in place but documented as input-mixing-required before re-enable. |
| B4 | **Layer-1 step-0 probe could never run AND could never fail.** (a) `sample_batch['image']` — `collate_fn` returns a `(images, targets)` *tuple*, so the probe raised `TypeError` on every run; (b) the probe's own `raise RuntimeError` was inside its own `except Exception` → downgraded to a warning; (c) raw uint8 input (no ImageNet normalization) would have distorted the very magnitudes being measured. | `src/training/train.py:2635-2659` | Rewritten: tuple unpacking, `_prepare_images(...)` normalization, empty-loader → explicit crash, the assertion raised *outside* the try, probe-infrastructure failure also crashes (a guard that silently no-ops is how RC-25 survived three rounds), `model.train()` in `finally`. |
| B5 | **Layer-2 guard evadable by NaN/inf.** `losses._s()` maps non-finite values to `0.0` (losses.py:1382-1390), so `det_cls = inf` would *pass* `cls_loss < 1e4`. | `src/training/train.py:1098` | Guard now also fails when the total loss tensor is non-finite at step 0. |
| B6 | **D7/D8/D9 had never executed.** All three call `POPWMultiTaskModel(num_det_classes=..., num_act_classes=..., num_psr_classes=..., backbone_type=C.BACKBONE_TYPE...)` — none of these kwargs exist (real signature: `pretrained, backbone_type, use_headpose_film, use_hand_film, use_videomae, train_pose`) → `TypeError` at construction. D7 additionally crashed at import (`from data import create_dataloaders` — symbol doesn't exist) and all three used `batch['image']` on a tuple. **Consequence: RC-25 has not actually been measured yet.** | `code/diag_feature_magnitude.py:32,98-104,116-126`; `code/diag_step0_logits.py:42-48,71-77`; `code/diag_weight_norms.py:61-80` | Constructors fixed to the real signature (`use_videomae=False` for light probes); dataset access via `IndustRealMultiTaskDataset` + tuple collate + ImageNet normalization; lazy imports with informative fallbacks. |
| B7 | **Wrong diagnostic control baseline.** D7/D9 built the "fresh" reference with `pretrained=False` — *random* init, not "fresh ImageNet-init" as labeled. ImageNet backbone weight norms/feature scales differ systematically from random init, so every backbone ratio would be inflated → false EXPLODED flags. D9 also scanned only `Conv2d`, missing most of the ConvNeXt backbone (block MLPs are `nn.Linear`). | `code/diag_feature_magnitude.py:103`; `code/diag_weight_norms.py:66,44-50` | Controls now `pretrained=True`; D9 scans `Conv2d` + `Linear`. |
| B8 | **FIX-4 (PSR expand bug) absent from this tree.** The sequence path computed ONE prediction (`encoded[:, -1, :]`) and `.expand()`ed it across all T frames — per-frame focal loss vs T different labels (optimum = constant window-average = collapse pressure) and identically-zero temporal-smooth gradient. This tree is an older lineage than the round-2 snapshot: it also lacked P5 and P7 (below). | `src/models/model.py:1850-1858` | Per-position predictions reapplied (`encoded.reshape(B*T, hidden)` → heads). Shape contract verified: forward still returns `[BT, 11]`; train.py:971 `view(B,T,-1)` intact. |
| B9 | **P5 (attention scale) absent.** `attn = qk^T / scale` with `scale = d^-0.5` *multiplies* logits by √d=8 (64× standard) → softmax saturation in the activity ViT. | `src/models/model.py:1095-1096` | `* scale`. |
| B10 | **P7 (det_conf raw logits) absent.** `det_conf = cls_preds.max(dim=1)[0]` — unbounded raw logits into the activity head (measured constant L2 243.39±0.001 under collapse). | `src/models/model.py:1807` | `torch.sigmoid(cls_preds.max(dim=1)[0])` (monotonic — ranking unchanged, scale now [0,1]). Zero-gate unchanged after it. |
| B11 | **Recovery checkpoints would be EMA blends.** `best.pth` saves EMA weights (train.py:3342-3347); the recovery preset didn't set `use_ema`, so a short recovery run's best.pth = EMA lagging toward init. Also a train/eval mismatch hazard: a checkpoint trained with `zero_det_conf=True` must be *evaluated* with it too, but eval scripts never apply presets. | `src/config.py` PRESETS; `src/models/model.py:1819` | `'use_ema': False` added to the recovery preset (+ `apply_preset` handling for `USE_EMA`/`USE_MIXUP`); `ZERO_DET_CONF_FOR_RECOVERY` now honors env `ZERO_DET_CONF=1` so eval runs can match training. |
| B12 | Cosmetics: `EFFECTIVE_BATCH` stale after presets; `--preset` help said "no-op on IndustReal". | `src/config.py:270`; `src/training/train.py:3510` | Recomputed in `apply_preset`; help text fixed. |

---

## 2. Answers to the specific questions

**Q1 (trace v2 plan → code):** Done above. Of the v2 prescriptions: FPN reinit
✓ correct, GroupNorm ✓ correct, EMA re-anchor ✓ correct, Kendall reset ✓,
`--reinit-heads` ✓, in-loop assertion ✓ (after B5 hardening). The pre-training
assertion (B4), recovery preset (B1/B11), and all three diagnostics (B6/B7)
were broken as uploaded.

**Q2 (`_reinit_dead_heads`, train.py:1654-1795):** **Correct.** Verified:
FPN 8/8 modules Kaiming-uniform a=1 + zero bias with `assert fpn_reinit == 8`
(1668-1684); det `cls_score` pi=0.05 → bias −2.944, std 0.01 (1690-1695);
`reg_pred` std 0.01, bias 0 (1696-1700); `cls_subnet`/`reg_subnet` correct
names, Kaiming-normal fan_out (1702-1711); act `proj_features` std 0.02,
`cls_token` trunc-normal 0.02, classifier std 0.01 bias −0.5, vit Xavier + LN
reset, tcn Kaiming (1714-1760); psr `per_frame_mlp` std 0.02, `output_heads`
std 0.01 bias −0.2 (1762-1781). Two non-issues worth knowing: the
`gap_p3/p4/p5` branch (1782-1789) is a structural no-op (those are
`AdaptiveAvgPool2d`, not Conv2d — harmless); and the reinit prior pi=0.05
differs from the constructor's pi=0.03 (model.py:527) — cosmetic, the reinit
value governs recovery runs.

**Q3 (step-0 assertion coverage):** `loss_dict['det_cls']` is populated
whenever `train_det=True` (losses.py:1394) and is `0.0` when detection is
disabled — the guard then passes trivially, which is correct behavior. The
real coverage holes were B4 (the pre-training layer never executed and
couldn't fail) and B5 (`_s()` sanitizes inf→0.0, evading `>= 1e4`). Both
fixed. Remaining accepted gap: if `det_cls` itself was NaN-sanitized to 0.0
while the *total* stays finite, the guard passes — the total-finiteness check
plus the Layer-1 probe (which reads raw logits, pre-loss) covers this in
practice.

**Q4 (GroupNorm placement):** **Your placement is correct** —
`Conv → GroupNorm → ReLU` (model.py:502-504) is the standard pre-activation
ordering (normalize the linear output, then rectify). GN *after* ReLU would
normalize a half-rectified distribution and re-introduce negative values into
what downstream expects to be post-ReLU features. `GroupNorm(8, 256)` = 32
channels/group, fine. One side effect to know, not fix: GN insertion shifts
the `Sequential` indices, so an old checkpoint's `cls_subnet.{2,4,6}.weight`
conv weights no longer map (index 2 is now ReLU, 4 is GN, ...) —
`_load_model_compat` will silently skip them and those convs stay at fresh
init. Irrelevant under `--reinit-heads` (they're re-initialized anyway), but
**a plain `--resume` of a pre-GN checkpoint now loads only the first subnet
conv per tower** — don't plain-resume old checkpoints into this model.

**Q5 (`apply_preset`):** The function body handled the three new globals
correctly — but on the wrong module object (B1), so the entire preset was a
no-op. Missing keys found and added: `use_mixup`, `use_ema`;
`EFFECTIVE_BATCH` recompute. After the fix, runtime-verified:
`apply_preset('recovery')` → `ZERO_DET_CONF=True, MIXED_PRECISION=False,
STAGED_TRAINING=False, USE_EMA=False, USE_MIXUP=False`, visible through both
`import config` and `from src import config`.

**Q6 (EMA re-anchor):** **Correct as implemented** (train.py:2611-2623):
clones current (post-reinit) param data into the shadow for
`det/act/psr/fpn` prefixes, and the order is right — the checkpoint's
`ema_shadow` restore (2498-2507) happens *before* the reinit block, so the
re-anchor wins for the affected tensors while the backbone shadow correctly
keeps the checkpoint's EMA values. Rebuilding EMA from scratch is not needed
(it would discard valid backbone EMA state). Note the
`(det_head.|detection_head.)` prefix pair is fine (only `detection_head`
exists; the extra prefix is harmless). With B11's `use_ema: False` in the
recovery preset, this path is now belt-and-suspenders for recovery runs and
the real protection for future long runs.

**Q7 (edge cases):**
- `cls_preds` absent at step 0 → Layer-2 logs a warning and skips the logit
  check (acceptable: the `cls_loss` check still runs); Layer-1 now treats it
  as a probe failure and crashes (correct — det head must exist on a recovery
  run).
- `loss_dict` missing `det_cls`/`cls` → `get(..., 0.0)` passes trivially;
  only possible when `train_det=False`, which is a legitimate ablation.
- Empty dataloader → Layer-1 now raises a clear `RuntimeError`
  ("train_loader is EMPTY") instead of silently warning; Layer-2 never runs
  (no steps), and the epoch loop proceeds to val — pre-existing behavior,
  acceptable given Layer-1 crashes first on recovery runs.

**Q8:** See verdict above. **Safe to run after this branch's fixes**, with §5
caveats.

---

## 3. Verified-correct list (no action needed)

- FIX-3 (dim==5 sequence gating) present (model.py:1731).
- 2026-06-07 head-pose-FiLM ordering hot-fix present (head_pose computed
  before `activity_proj`; modulated `c5_mod` feeds the activity head).
- losses.py has the LDAM `_fit_to_width` re-alignment and the corrected
  temporal-smooth sign (`pred_change - label_change`).
- `evaluate_all` passes `clip_rgb` to the model (evaluate.py:2842/2848) — no
  VideoMAE zero-feeding in the main eval loop.
- `criterion = MultiTaskLoss(num_classes_act=C.NUM_CLASSES_ACT)` ✓ (75).
- `_REINIT_HEADS_ACTIVE` module flag + `global` declaration ✓.

---

## 4. Residual risks (documented, not fixed here)

1. **The `data/` package is not in this repository** — the dataset/collate
   side was audited against the round-2 snapshot copy
   (`opus_consult_2026_06_10_v2/code/industreal_dataset.py`), which returns
   `(images, targets)` tuples. If the live `data/` module diverges, re-check
   the probe unpacking and the `clip_rgb` key.
2. **This `src/` tree is an older lineage than the round-2 snapshot**: it
   lacked FIX-4/P5/P7 (now reapplied) and has no `__nan_detected__` sentinel
   plumbing in losses.py. Diff the live machine's tree against this branch
   before running — there may be other silent reversions.
3. **FeatureBank still dead** (`video_ids` never passed at train.py:1077 or
   evaluate.py:2842) — accepted deferral (P11); activity remains effectively
   per-frame until this is wired.
4. **Old-checkpoint subnet weights partially non-loadable** post-GroupNorm
   (see Q4) — only matters for plain resumes, not recovery.
5. The Layer-1 probe consumes `next(iter(train_loader))` — with
   `WeightedRandomSampler` this draws one batch and discards it; harmless,
   but the probe adds one loader spin-up to startup.

---

## 5. How to launch the recovery run (after this branch)

```bash
# 0) Zero-GPU diagnostics first — these can now actually execute:
CHECKPOINT=<...>/latest.pth python code/diag_weight_norms.py        # D9: locate blast radius
CHECKPOINT=<...>/latest.pth python code/diag_feature_magnitude.py   # D7: confirm RC-25
CHECKPOINT=<...>/latest.pth REINIT_FPN=1 python code/diag_step0_logits.py  # D8: verify fix

# 1) Recovery training (preset now actually applies):
python src/training/train.py --preset recovery --reinit-heads \
    --resume <...>/latest.pth --subset-ratio 0.25 --max-epochs <start+3> --seed 42
# Startup log MUST show: MIXED_PRECISION=False, STAGED_TRAINING=False,
# ZERO_DET_CONF_FOR_RECOVERY=True, USE_EMA=False, USE_MIXUP=False
# and "[STEP-0 ASSERT] PASSED". If D9 shows backbone blowup → fresh ImageNet
# init instead of --resume (Branch B from Answer v2).

# 2) Evaluating a recovery checkpoint (train/eval consistency):
ZERO_DET_CONF=1 EVAL_CKPT=<...>/best.pth python <eval entrypoint>
```

Success gates unchanged from Answer v2: step-0 cls_loss in O(10²–10³), not
10⁷; after the run, det probe shows `bestIoU > 0.5` matches appearing,
activity predicts ≥ 4 classes, PSR ≥ 3 unique patterns. Then scale data.
