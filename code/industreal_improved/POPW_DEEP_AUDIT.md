# POPW Deep Audit — 2026-06-10

Every claim below was verified against the uploaded source files (train.py 3733 lines, model.py 2167, losses.py 1505, evaluate.py 4004, config.py 699, industreal_dataset.py 1383, eval_post_reinit.py, metrics.py) and the evidence files (train_digest.log, metrics_baseline_pre_reinit.json). Line numbers refer to the uploaded snapshot; your live tree at `/media/newadmin/master/POPW/...` may have drifted, which is why `apply_popw_fixes.py` patches by exact-string match and refuses anything that doesn't match verbatim.

---

## 0. URGENT: the recovery retrain is already dead

`train_digest.log` contains the same traceback **twice**:

```
train.py:3196  _check_per_class_activity_sanity(val_metrics, epoch)
train.py:2002  report_per_class_accuracy(per_class_acc, C.ACT_CLASS_NAMES, k=5)
evaluate.py:869  row_sums = cm.sum(axis=1).clip(min=1.0)
numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1
```

Mechanism, verified:

- `_check_per_class_activity_sanity` (train.py:1986–2003) fires when `epoch % 10 == 0` and epoch ≥ STAGE1+STAGE2 (15). You resumed from **epoch 40** → it fires on the *first* epoch of the retrain.
- It passes `val_metrics['act_per_class_acc']` — a **1-D** per-class-accuracy list (built at evaluate.py:805–807, returned at :850) — into `report_per_class_accuracy` (evaluate.py:862), which expects a **2-D confusion matrix** and calls `cm.sum(axis=1)` → AxisError.
- The exception is re-raised by the val retry logic (train.py:3236 `raise` — non-OOM, no "empty" in message) and propagates to top level, killing the process.
- Checkpoint ordering (train.py:3441–3460): `latest.pth` and the `epoch_{N}_end` crash-recovery save happen **after** validation. So each crash loses the entire epoch of training. The retrain (PID 2416305) cannot get past epoch 40 until this is fixed.

**Fix:** FIX-1 + FIX-2 in `apply_popw_fixes.py` — pass `act_confusion_matrix`, harden `report_per_class_accuracy` for 1-D input, and wrap the call in try/except so a logging helper can never abort training again.

---

## 1. Root-cause finding: fake-sequence regrouping of ordinary batches

This is, in my assessment, the single most important discovery in the audit, and it directly explains the PSR collapse signature (`unique_binary_patterns=1`).

**model.py:1849–1855 + 1972–1976.** `forward()` decides sequence-ness from the persistent tag `model._seq_len`, which train.py:2355 sets to **4 permanently** whenever `USE_PSR_SEQUENCE_MODE=True`. The gate is:

```python
B_main = BT // seq_len if seq_len > 1 else B
T_main = seq_len if seq_len > 1 else 1
if T_main > 1 and B_main * T_main == BT:   # sequence path
```

For a **normal 4-D batch** of B independent frames, `BT = B`. Whenever B is divisible by 4, the check passes and the model treats unrelated frames as one temporal window:

| Context | Batch | Effect |
|---|---|---|
| Train, batch=4 (current retrain default) | BT=4 → B_main=1, T_main=4 | every normal batch becomes 1 fake 4-frame sequence |
| Train after OOM auto-reduce, batch=2 | 2//4=0 → check fails | accidentally dodges the bug |
| In-training validation, VAL_BATCH_SIZE=16 | B_main=4, T_main=4 | 16 independent val frames → 4 fake sequences |
| `eval_post_reinit.py`, EVAL_BS=4 | B_main=1, T_main=4 | the baseline JSON was produced under this bug |

**Then the second half of the bug (model.py:1997–2007):** in the sequence path the causal transformer's output is reduced to `encoded[:, -1, :]` (last position only), the 11 heads produce one prediction, and it is `.expand()`ed across all T positions and reshaped to `[BT, 11]`. Consequences, all verified in the loss code:

1. The per-frame focal loss compares **one prediction against T different labels**. The loss-optimal output is the window-average label — i.e. the marginal component prevalence — which after thresholding is precisely the **all-zeros constant pattern** you observed. The head was not failing to learn; it was learning the optimum of a mis-specified objective.
2. The temporal-smoothness loss (losses.py:1222–1244) computes `p_i[1:] - p_i[:-1]` — **differences of the same tensor — exactly zero with identically zero gradient**. It contributes a label-dependent *constant* to `loss_psr`, which still feeds Kendall and inflates `log_var_psr` for nothing.
3. Every PSR eval number ever produced with `USE_PSR_SEQUENCE_MODE=True` (including `metrics_baseline_pre_reinit.json`) measured predictions that are constant within groups of 4 **by construction**.

**Fixes:** FIX-3 (gate the sequence path on `images.dim()==5` only, read T from the tensor) and FIX-4 (apply the output heads to *every* encoded position — causal masking already guarantees position t only sees frames ≤ t, so each position is a valid per-frame prediction, and the smooth loss finally gets a real gradient).

After these two fixes, normal batches take the T=1 feature-bank path consistently in train and eval, and the dedicated sequence batches genuinely train the causal transformer per-frame.

---

## 2. Claims in the six MD files that do not match the code

You asked me to verify the documents against the sources before they go to "Opus". These are the discrepancies that matter; sending the MDs uncorrected would have Opus solving the wrong problem.

**2.1 The "detach seq-mode PSR output" fix does not exist.** `03_CURRENT_RECOVERY.md` (fix #3, "train.py:~1660") and `05_MASTER_PROMPT.md` (§3 item 3) describe detaching the sequence-mode PSR output. There is **no `.detach()` anywhere on the seq path** — the only detach calls in train.py are checkpoint serialization (lines 681–733) and an EMA clone (2612). Line 1660 falls inside `_reinit_dead_heads`. The seq branch calls `scaler.scale(loss_seq).backward()` normally (train.py:1058). Whatever stabilized the retrain, it was not this fix, because it was never written.

**2.2 The bf16 NaN mechanism story is unsupported.** `02_COLLAPSE_CRISIS.md` claims `retain_graph=True` was "implicit in our pipeline" and that the seq batch's autograd graph poisons the next batch. There is no `retain_graph` anywhere in the codebase, and PyTorch does not retain graphs across separate forward/backward cycles. The actually plausible cross-batch contamination channel is in the code: the seq branch mutates `criterion.train_det/pose/act/psr` and **fails to restore them on two of its three early-`continue` paths** (train.py:1043–1057) — one NaN-skipped seq batch and every subsequent batch trains PSR only, with det/pose/act losses zeroed, until the run dies. FP32 may still be the right empirical call, but the MD's causal narrative should be rewritten. (FIX-5 closes the leak.)

**2.3 FOCAL_ALPHA is 0.75, not 0.25.** `01_WHAT_WE_BUILT.md` says detection uses Focal(α=0.25, γ=2). config.py:343 sets `FOCAL_ALPHA = 0.75` ("positives get 3× weight"). With α=0.75 the *negatives* are down-weighted to 0.25 — a sustained push toward higher scores, which is a credible contributor to the cls-score saturation at 1.0 (the documented bias-overshoot). Compounding it: once the head saturates, the negative-side corrective loss is huge, but the smooth cap `DET_LOSS_CAP=50` throttles its gradient by `cap/x` (losses.py `_smooth_cap`) — at loss 10,000 the corrective gradient is scaled by 0.005, and Kendall simultaneously inflates `log_var_det` to down-weight it. The caps create gradient starvation exactly when the head most needs correction. `02_COLLAPSE_CRISIS.md` Bug B says caps don't help because collapsed loss is *low* — true for activity/PSR, **backwards for detection saturation**, where the loss is high and the cap suppresses the recovery signal.

**2.4 Activity is 75 classes, not 74.** config.py pins `NUM_CLASSES_ACT = 75` (NA + raw IDs 1–74) with an assertion; the MDs repeatedly say 74 and `Linear(512→74)`. The 02 confusion-matrix excerpt also conflates "all predictions land on class 27" with seven different classes each listed as receiving all predictions — internally inconsistent as written.

**2.5 The activity reinit bias=-0.5 is a softmax no-op.** `_reinit_dead_heads` (train.py:1664) sets a *uniform* −0.5 bias on all 75 logits and the MD claims it "forces the no-class baseline". Softmax is shift-invariant: a constant added to every logit changes nothing. model.py:1302–1306 itself documents this exact fact for the old +6.0 init. The real benefit of the activity reinit is the weight re-init (std=0.01) plus full proj/ViT/TCN reset; the bias claim should be deleted from the MDs. (For PSR and detection the biases are per-sigmoid and the reinit reasoning is valid.)

**2.6 "Class-balanced sampling is not running" is false.** `03_CURRENT_RECOVERY.md` open problem #1 and H5 both say `WeightedRandomSampler` isn't in use. It is: `_build_loader` (train.py:227) calls `ds.get_sampler()` for the train split, and `get_sampler` (industreal_dataset.py:1172–1193) returns a class-balanced `WeightedRandomSampler` with effective-number weighting. H5's experiment B ("add the sampler") is partially already the baseline.

**2.7 VideoMAE never receives input — at all.** Stronger than H4's suspicion: the global collate is `collate_fn_sequences` (train.py:104, because `USE_PSR_SEQUENCE_MODE=True`), and that collate **does not include `clip_rgb`** in targets (industreal_dataset.py:1299–1383; only the per-frame `collate_fn` at :1292 stacks it). So `targets.get('clip_rgb')` is always None in training *and* evaluation, `videomae_feat=None`, and ActivityHead pads with `torch.cat([feat, torch.zeros_like(feat)])` (model.py:1350) — **half the classifier's input is permanently zero**, while the frozen 22M-param stream burns ~600 MB VRAM doing nothing.

**2.8 The temporal stack is temporally blind in training.** FeatureBank (model.py:1148–1151): with `video_ids=None` — which is what every training and eval forward passes — the bank is **the current frame replicated 16×**. The TCN and both ViT blocks therefore process constant sequences; PoseFiLM-conditioned "temporal" activity recognition is in practice per-frame. The PSR causal transformer is likewise either fed replicated banks (T=1 path) or, until FIX-3/4, fake sequences. The architecture-as-trained is a per-frame model carrying a large amount of dead temporal machinery. This reframes H2/H3/H4 considerably.

**2.9 Body pose is never trained, so PoseFiLM conditions on noise.** `loss_pose` only fires when `'keypoints' in targets` (losses.py:1095), and neither collate emits keypoints — IndustReal has no keypoint GT (the MultiTaskLoss comment acknowledges this). The PoseHead output feeding PoseFiLM is therefore an untrained network's output: structured noise modulating C5 for the activity branch. The `position_MAE_mm=739` in the MDs belongs to the *head-pose* head; "body pose survived" in 02 is not supported — body pose was never in the loss.

**2.10 Smaller real bugs found along the way.**
- `eval_post_reinit.py` reads `getattr(C, 'USE_HEADPOSE_FIM', False)` — typo, missing L — so every re-eval builds the model **without** headpose_film while the checkpoint contains it (weights silently dropped via `strict=False`). FIX-8.
- `_reinit_dead_heads` block #4 (Kendall log_var reset) is dead code: it checks `model.criterion` (the model has no such attribute) and `log_var` (the params are `log_var_det` etc.). The MD table listing Kendall reset as part of reinit is aspirational.
- losses.py `_safe()` sets `_nan_detected_this_step = True` **without `nonlocal`** — a dead local write, so the final NaN guard never raises `__nan_detected__` and train.py backwards through NaN-replaced losses it was designed to skip. FIX-6.
- The temporal-smooth loss uses `diff_l = -1 * (label diff)` — it pushes the predicted transition in the **opposite** direction of the label transition. Dormant only because the expand bug zeroed its gradient; it becomes actively harmful after FIX-4. FIX-7.
- The NaN-guard loop at losses.py:1046–1080 runs **before** `loss_pose/act/psr/head_pose` are computed (they're still `zero` at that point), so it only ever guards `loss_det`. Redundant with the later guards, but the comment block claiming it guards "ALL individual losses" is wrong.
- model.py pseudo-keypoint block (1875–1932): `top_conf.argmax()` indexes a `[B, 24]` class tensor into the `[A, 4]` anchor array — index-space mismatch, and the same index is reused for every batch item. Currently dead (`train_pose=True`), but should not be revived as-is.

---

## 3. The eight fixes (applied by `apply_popw_fixes.py`)

| ID | Pri | File | One-line description |
|---|---|---|---|
| FIX-1 | P0 | train.py | sanity check passes confusion matrix, wrapped in try/except — unblocks the retrain |
| FIX-2 | P0 | evaluate.py | `report_per_class_accuracy` accepts 1-D input |
| FIX-3 | P0 | model.py | sequence path gated on `images.dim()==5`; no more fake sequences from 4-D batches |
| FIX-4 | P0 | model.py | per-position PSR predictions in seq mode (kills the expand-copy objective) |
| FIX-5 | P1 | train.py | restore criterion train_* flags on both leaking seq-branch continue paths |
| FIX-6 | P1 | losses.py | `nonlocal` in `_safe()` so `__nan_detected__` actually reports |
| FIX-7 | P1 | losses.py | temporal-smooth label-change sign corrected |
| FIX-8 | P1 | eval_post_reinit.py | USE_HEADPOSE_FIM typo |

Self-test passed: all 8 anchors found exactly the expected number of times in the uploaded snapshot, and every patched file parses. Run order on the workstation:

```bash
# 1. stop the (crashed/looping) retrain if still alive
kill 2416305 2>/dev/null

# 2. dry run first — anything that prints [SKIP] has drifted; patch those by hand
python3 apply_popw_fixes.py --root /media/newadmin/master/POPW/working/code/industreal_improved

# 3. apply (originals saved as *.bak_prefix_20260610)
python3 apply_popw_fixes.py --root /media/newadmin/master/POPW/working/code/industreal_improved --apply

# 4. restart the retrain from the reinit checkpoint as before
```

Note: because FIX-3 changes which code path PSR takes at batch=4 (feature-bank path instead of fake-sequence path), the loss trajectory will look different from the first 8 minutes of the previous retrain. That difference is the bug being removed, not a regression.

---

## 4. Hypothesis verdicts (H1–H8), grounded in what the code actually does

**H3 — PSR causal transformer: fix first, then decide; the current evidence is void.** The transformer has never been trained or evaluated as a temporal model: T=1 batches feed it replicated banks, sequence batches fed it fake or copy-output sequences, and eval grouped unrelated frames. With FIX-3/4 the seq batches finally provide genuine per-frame causal supervision at near-zero extra cost. Run 5 epochs post-fix; if `psr_overall_f1` is still ~0 then drop the transformer with a clear conscience — but dropping it *before* the fix would discard an untested component based on contaminated evidence.

**H4 — VideoMAE: drop it now.** Not a hypothesis anymore — §2.7 proves it receives no input in the current configuration while costing ~600 MB VRAM, ~22M params and half the activity classifier's input width. Set `USE_VIDEOMAE=False`. This is also your cheapest VRAM unlock (likely batch=4 stable in FP32), which weakens the case for H6.

**H1 — Kendall vs GradNorm/PCGrad: keep Kendall+clamp, defer the swap.** The observed head-silencing was driven by identifiable bugs that pollute the loss signal Kendall balances: the seq-mode constant smooth-loss term inflating `log_var_psr` (§1.2), the flag-leak making the loss composition flip mid-run (§2.2), and the caps throttling detection's recovery gradient (§2.3). Swapping the balancer before removing these would attribute their damage to Kendall. Re-measure the per-head gradient-norm ratio after the fixes; only if it's still pathological is GradNorm worth a day.

**H2 — staged vs joint: the stage-3 collapse evidence is contaminated; don't re-architect yet.** Stage-3 entry coincided with the activation of the two heads whose objectives were broken (fake-seq PSR, LDAM-DRW-at-epoch-0 activity on ~40 samples/class). The cleanest reading is "stage 3 turned on the buggy losses", not "staging is the wrong inductive bias". Keep the curriculum for the post-fix run; revisit only if collapse recurs with clean losses.

**H5 — activity loss: the highest-value remaining ablation.** The sampler half of H5 is already running (§2.6), so the experiment reduces to one toggle: `USE_LDAM_DRW=False` falls back to the existing `ClassBalancedFocalLoss` (γ already softened to 1.0, label smoothing 0.1). LDAM margins of `max_m=0.5, s=30` estimated from ~40 samples/class, with DRW re-weighting from step one, on top of an *already class-balanced sampler* (double-correcting the imbalance) is a lot of machinery fighting itself. I'd run A=current vs B=`USE_LDAM_DRW=False` as your first post-fix ablation.

**H6 — smaller backbone: defer.** The collapse causes found are head/loss/plumbing bugs, not backbone capacity, and the features are verified alive. Dropping VideoMAE buys most of the VRAM H6 was after. If you still want batch=8, freezing ConvNeXt stages 1–2 is a far smaller diff than an FPN-channel rework for EfficientNet-B0.

**H7 — FiLM: disable hand-FiLM for now.** §2.9 shows PoseFiLM is conditioned on an untrained pose head — i.e., learned noise injection into the activity branch. Set `USE_HAND_FILM=False` until there's a meaningful pose signal (or condition on `det_conf`, which at least carries trained information once detection recovers). HeadPoseFiLM conditions on a trained head and can stay.

**H8 — three partial backbones: no.** The single backbone was never shown to be the problem; the failures localize to head objectives and batch plumbing. Tripling backbones on a 12 GB card to fix loss bugs would be solving the wrong layer of the stack.

**Config changes I'd make alongside the patches** (one-liners, your call, not auto-patched): `USE_VIDEOMAE=False` (proven dead); `FOCAL_ALPHA` back toward 0.25–0.5 (0.75 plausibly fed the saturation, and the reinit's pi=0.05 already gives positives a healthier start); consider raising `DET_LOSS_CAP` (e.g. 50→200) so a saturating head can be pulled back before the cap throttles its corrective gradient; and `USE_HAND_FILM=False` per H7. If you change `USE_VIDEOMAE`, remember the classifier input width halves — the activity head must be re-initialized, which `--reinit-heads` already does.

---

## 5. Suggested 7-day sequencing (replaces the plan in 05_MASTER_PROMPT)

Day 1: apply the 8 patches, set `USE_VIDEOMAE=False` + `USE_HAND_FILM=False`, restart the reinit retrain (batch=4 should now hold in FP32). Day 2: re-eval; this is the first eval in the project's history where PSR numbers are not artifacts, so treat it as the new baseline. Days 3–4: H5 ablation (`USE_LDAM_DRW` on/off). Day 5: detection focal sweep (α ∈ {0.25, 0.5, 0.75} now that the cap/alpha interaction is understood). Days 6–7: H3 decision on the causal transformer based on clean evidence, then scale subset 5%→10%. Correct the MD files (§2 items) before any handoff — particularly the nonexistent detach fix and the retain_graph narrative, which would misdirect any reviewer.

The one-sentence summary: the architecture was never given a fair chance — the PSR objective was structurally mis-specified by the fake-sequence/expand bugs, the activity branch was half-zeros with noise conditioning, and the watchdog that was supposed to protect the retrain is the thing currently killing it. Fix the plumbing first; the architectural questions get much smaller afterwards.
