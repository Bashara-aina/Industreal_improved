# 102 — Round 5: Answers to the 20 Questions (file 100), GPU Crisis Playbook & New Fixes (F19–F21)

**Date:** 2026-07-03
**Reviewer:** Claude (Fable) — verified against `origin/main` (the code your training PID runs) plus the rebased consultation branch.
**Branch:** `claude/rf4-architecture-consultation-5mnnu5`, rebased onto current main.

---

## ⚠️ READ FIRST — three things that change how you read files 97–100

### 1. Your audit-agent files fed file 100 facts from a DIFFERENT codebase
Several question premises in file 100 are **false for the code that is actually
training**, and the pattern (plus the `industreal_improved_to_archive` paths that
appear in `src/test_eval_fix.py` / `eval_post_reinit.py`) says your local
`agent*.md` audit files described an old/archived working copy:

| File-100 claim | Reality in the training code |
|---|---|
| "Pose-Derived Detection (PDD), not a learned detection head" (Q5) | Detection is a **learned RetinaNet-style head** (5.3M params, cls+reg on FPN). mAP is measured from its predictions. No PDD in the model path. |
| "ConvNeXt-Base + MViTv2 temporal encoder" (Q4, appendix) | **ConvNeXt-Tiny**; MViTv2 appears nowhere in the model. |
| "PSR_SEQUENCE_LENGTH=2" (Q9/Q16) | **8** (config line ~1054). |
| "`clip_grad_norm_(max_norm=1.0)`" (Q2/Q15) | **GRAD_CLIP_NORM=5.0**. |
| "USE_PSR_TRANSITION=False, PSR trains per-frame" (Q9/Q15/Q16) | **True** — PSR trains on transition targets over 8-frame sequences. |
| "CB-Focal (beta=0.999) active for activity" (file 98 §2.1) | **USE_CB_FOCAL_ACT=False** — activity uses plain CE + label smoothing 0.1 (good: no double balancing with the balanced sampler). |
| "Kendall formula has a 0.5 factor; equilibrium exp(-lv)·L=2" (Q13) | This implementation is `exp(-lv)·L + lv` — **no 0.5**; equilibrium is `lv* = ln(L)`. |

The training process itself runs the right code — the `[KENDALL step=…]` lines in
your logs are the F2 sentinel format, which only exists post-merge. It's the
*audit documents* that are contaminated. Recommendation: regenerate any agent
audit against the repo the PID actually runs (`readlink /proc/<pid>/cwd`).

### 2. main does NOT have F17/F18 — merge this branch before the next restart
PR #20 merged rounds 1–2 (F1–F16) only. Still missing from main:
- **F18 (activity double-ramp)**: on main, warmup activity supervision is
  **ramp²** — the current run's epochs 0–2 ran at 11%/44%/100% of intended
  weight instead of 33%/67%/100%. This directly depresses the epoch-2
  activity numbers you're judging viability from (macro-F1 0.006 was measured
  on a head that had received far less supervision than believed).
- **F17** (`src/data/__init__.py` + root-anchored `.gitignore`): without it a
  fresh clone cannot run training — matters for the paper's code release.
- Plus this round's F19–F21 and the 22-test regression suite.

### 3. The combined metric's pose term uses RAW MAE, not degrees — combined=0.168 is 81% pose
`_compute_combined_metric` receives `head_pose_MAE` = the **raw L1 error on the
standardized 9-dim vector** (~0.10 for your converged pose), so the pose term is
`0.15·1/(1+0.10) ≈ 0.136` — nearly saturated (max 0.15) and contributing **81%
of the entire epoch-2 combined=0.168**. Q14's hand-calculation (0.054) assumed
degrees; both it and doc 96's "combined > 0.25 at epoch 5" threshold were
mis-calibrated. **F20** now logs `combined_v2` (degrees-normalized pose term,
diagnostic only — selection still uses v1 so best-metric history stays
comparable). **Use the per-head gates in §GPU-free thresholds below, not any
single combined number.**

---

## Section 1 — GPU Stability Crisis (Q1–Q4, Q17, Q18): the playbook

**First, disambiguate the two watchdogs** (Q2 conflates them):
- The **CUDA launch-timeout watchdog** is a *per-kernel* limit (~seconds) that
  the driver arms **only on GPUs with an active display/X screen**. It fires as
  `cudaErrorLaunchTimeout` and resets the GPU.
- Your **Python watchdog (1800 s)** is wall-clock heartbeat staleness. A slow
  `clip_grad_norm_` could never trip the CUDA watchdog through accumulated wall
  time — only a single long-running kernel can. The Q2 causal chain
  (loss spike → slow norm over 30M params → 1800 s) mixes the two and is not
  the mechanism; norm reductions over 33M params are milliseconds on GPU.

**The decisive diagnostic (do this first, 10 minutes):**
1. At the next crash, read the **Xid code**: `sudo dmesg | grep -i xid` (or
   `journalctl -k | grep Xid`). Xid 8/31/59 (launch timeout / watchdog) vs
   Xid 79/109/119 (bus/driver faults) tells you definitively whether this is
   the display watchdog or a driver/hardware fault. Everything downstream
   branches on this.
2. Check whether X claimed the 5060 Ti: `grep -iE "NVIDIA|GPU" /var/log/Xorg.0.log`
   and `xrandr --listproviders`. Xorg **auto-adds every GPU** unless told not
   to — a headless card can still carry an X screen, which is exactly what
   arms the launch-timeout watchdog on it (Q3's intuition, simpler mechanism
   than PCIe TLP pressure).
3. If X claims the 5060 Ti → pin X to the 3060 in `/etc/X11/xorg.conf`:
   `Section "ServerFlags" Option "AutoAddGPU" "false" EndSection` plus an
   explicit `Device` section with the 3060's `BusID`. Or test instantly:
   `systemctl stop gdm` and run one epoch headless. If the timeout vanishes,
   this is the permanent fix and the crisis is over.

**Your mitigation set is partly pointed the wrong way (Q1):**
- `ALLOW_TF32=False` / `NVIDIA_TF32_OVERRIDE=0` makes every conv/matmul kernel
  **2–4× slower** → individual kernels get *closer* to a per-kernel timeout,
  not further. TF32 does not cause launch timeouts. **Re-enable TF32** (or
  better, turn on `MIXED_PRECISION=True` with the F6 bf16 path — same effect,
  bigger: the longest kernels shrink ~2×, which directly attacks the failure
  mode, Q4's answer is yes).
- `CUDNN_BENCHMARK=False` is correct to keep (autotune launches long trial
  kernels — a classic watchdog trigger).
- `TORCH_CUDNN_V8_API_DISABLED=1` is ambiguous (legacy kernels can be slower);
  keep only if evidence supports it.
- Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the "always around
  epoch 5, batch ~100" reproducibility smells like allocator state (the
  per-seq-batch `torch.cuda.empty_cache()` calls + epoch-end cache clears
  cause refragmentation churn early in later epochs).
- Enable `nvidia-persistenced` (cheap, removes driver re-init variance).

**To identify the exact op (Q4's last part):** resume from the epoch-5
checkpoint with `CUDA_LAUNCH_BLOCKING=1` for a ~200-batch diagnostic run only.
The 30–50% slowdown is irrelevant for 200 batches, and the crash then surfaces
as a Python exception with the exact line instead of an async death. This is
the single highest-information experiment available.

**Q17 (driver/CUDA versioning):** CUDA 13.0 + driver 595.71.05 + consumer
Blackwell is past my knowledge cutoff — I cannot confirm or deny a specific
known bug, and you should treat any claim otherwise as fabrication. The
version-management moves, in order of preference: (a) latest *production*
branch driver for Blackwell rather than the newest feature branch; (b) a
PyTorch build against CUDA 12.x runtime (drivers are backward-compatible with
older CUDA runtimes); (c) there is no Linux TdrDelay equivalent for consumer
cards — the watchdog goes away by removing the X screen, not by tuning it.

**Q18 (the 3060 escape hatch):** yes, and don't frame it as defeat. The 3060
(Ampere, 550-series driver) has never produced this failure. At 12 GB:
batch 2–3 × accum 8–6 (keep effective 16, 'auto' peak factor handles LR),
~2.5–3× slower per epoch but **P(reaching epoch 100) ≈ 1** vs your own 60%
estimate on the 5060 Ti. The optimal split while the Xorg diagnosis runs:
**main run on whichever card survives; ablation suite (F16) on the other.**
The ablations are single-task (lighter) and fit the 3060 comfortably. If both
the Xorg fix and driver moves fail, a week of cloud GPU (~$50–100) buys the
entire RF10 run and removes the single biggest threat to the paper timeline —
cheap insurance against a 40%-probability project failure.

---

## Section 2 — Detection & Activity (Q5–Q8)

**Q5 — premise false.** Detection is a learned head (see ⚠️ table). The RF2
reference (0.31 mAP50_pc @ epoch 21) was achieved *with the F1 bug active*, so
it is a **floor**, not a ceiling, for the fixed pipeline. Expected: 0.20–0.30
by epoch 12–15, 0.30–0.45 by epoch 25–35. The "theoretical max given pose MAE"
question dissolves — detection does not route through pose.

**Q6 — the rising activity loss is mostly healthy, with one number to watch.**
Mechanics: ramp completed (epoch 3) + LR still climbing + the head escaping
collapse (entropy 0→1.27, distinct 1→5) — CE goes *up* while a collapsed head
starts spreading probability mass before features become discriminative. The
number to watch: **act loss > ln(69)=4.23 sustained** (your epoch-4 tail shows
4.3–5.0) means confidently-wrong predictions on the balanced-sampled tail, not
mere diffusion. Verdict thresholds: if by **epoch 8–10** act loss isn't back
under ~4.0 with pred_distinct ≥ 15, the per-frame GAP feature is the binding
constraint → activate the Tier-2 temporal path (doc 96 §4) at RF6. Your
lv_act=0.114 rising toward its equilibrium `ln(0.8·L)≈1.0` is Kendall working
*correctly* (down-weighting a noisy task); it is not a harm signal, and while
lv_act sits below equilibrium, activity receives *more* than Kendall-optimal
precision — favoring exactly the head you need. Do not touch the weight.

**Q7 — premise mixup.** FOCAL_ALPHA applies to **detection**, not activity;
gamma_neg is 1.5, not 2; and negatives are OHEM-mined (top ~2:1 hardest per
positive). Corrected analysis: hard negatives are not starved — OHEM selects
precisely the negatives with high p, whose gradient weight `0.5·p^1.5` is
substantial (p=0.5 → 0.18, p=0.9 → 0.43). Easy negatives are silent by design.
The alpha=0.5 + gamma_pos=0 + OHEM combination is sound for 24-class detection
with extreme imbalance. Activity's loss is plain CE+LS and none of this applies.

**Q8 — 69 groups is what your data produces; don't re-group mid-run.** With
the balanced sampler, per-class exposure is equalized regardless of the
power-law. 768-dim features with 25K samples are not the separability
bottleneck (rule-of-thumb capacity is orders of magnitude above 69 classes) —
feature *quality* (per-frame GAP vs temporal) is. The grouping question is an
ablation, and the runner already has it (`grouping-none`); if you want a
threshold sweep (e.g., ACT_HYBRID_THRESHOLD 100→200 → fewer groups), that's a
30-minute offline count with `_act_grouping()`, not a training change.

---

## Section 3 — PSR & Pose (Q9–Q12)

**Q9 — corrected: T=8, transition objective ON.** PSR's ~1,645 seq steps/epoch
train a ~3M-param head+transformer: expect first nonzero transition-F1 at
epochs 8–12, not epoch 5. The 15× pre-Kendall amplification only circulates
inside the detached PSR branch (backbone never sees it), and Kendall is
pinned at the MAX_PSR=0 ceiling exactly as doc 96 predicted (your measured
lv_psr −0.014 ≈ 0). Flip `detach_psr_fpn=False` at RF6+ **only if** PSR F1@±3
plateaus < 0.45 — with the F1 snapshot-restore path this is now safe to try.

**Q10 — your instinct is right and the codebase already agrees.** Transition
targets + `psr_f1_at_t` (±3-frame, monotonic decoding) exist precisely because
fill-forward per-frame accuracy is gameable by all-ones. Treat binary
comp-acc (0.291) as a *liveness* indicator only; never report it as a result.
A random transition predictor's F1@±3 is ≈ 2·tol·(transitions/frames) ~ 0.01–
0.05 — quote that as the null baseline in the paper. Adding a
transition-frame-restricted accuracy metric is a good eval-only patch for RF6.

**Q11 — solved, and your fear is unfounded.** Precision is `exp(-lv)`; the cap
is `lv_used = max(lv_pose, lv_det.detach())`. Your raw `lv_pose=-1.000` is a
fossil restored from an old checkpoint (init was changed to 0.0, but resumes
carry the old value), and it never moves because `torch.maximum` passes zero
gradient to the smaller argument and F14 removed the weight decay that used to
drag it. The **effective** pose log-var is `lv_det` (0.075) → precision 0.93,
not 2.718. There is no hidden 2.7× pose amplification and no overshoot risk
(file 98 §3.5's "risk of over-performing" is moot). **F19** now logs
`lv_pose_EFFECTIVE` + a `HP_PREC_CAP ACTIVE` flag so this can't be misread again.

**Q12 — yes, but later and for a better reason.** Freezing pose head +
HeadPoseFiLM/PoseFiLM after convergence saves little compute (~1M params of a
33M graph) but *stabilizes pose through the peak-LR window* — the real
benefit. Criterion: fwd MAE stable within ±0.5° across 3 consecutive evals,
then freeze at RF6/RF7. Before epoch 15, don't.

---

## Section 4 — Multi-Task Balancing (Q13–Q16)

**Q13 — no 0.5 factor in this implementation; the system is closer to
equilibrium than you think.** Equilibrium is `lv* = ln(L)`: det loss ~1.0 →
lv*≈0, measured 0.075 — **at equilibrium, tracking a slowly-changing loss**,
not diverging. lv_act (0.114, equilibrium ≈1.0) drifts up slowly, which
transiently over-weights activity — beneficial. Do **not** give log_vars a
special LR mid-run; the slow adaptation is a feature (fast log_vars chase
batch noise). Revisit only if a task's lv is >1.5 from `ln(L)` for 5+ epochs.

**Q14 — resolved by the unit discovery (⚠️ §3).** Your 0.054 hand-calc assumed
degrees; the code feeds raw MAE, hence 0.168. Adopt **per-head gates** as the
real decision instrument (they were always the intent of doc 96 §2.7):

| Gate (post-restart epochs) | det mAP50_pc | act macro-F1 | act distinct | fwd MAE | PSR F1@±3 |
|---|---|---|---|---|---|
| Epoch 8 | ≥ 0.18 | ≥ 0.03 | ≥ 12 | < 13° | > 0 |
| Epoch 12–15 (peak LR) | ≥ 0.25 | ≥ 0.08 | ≥ 20 | < 12° | ≥ 0.10 |
| Epoch 25–35 | ≥ 0.32 | ≥ 0.13 | ≥ 30 | < 11° | ≥ 0.25 |

Keep v1 combined for checkpoint continuity; read `combined_v2` (F20) for an
honest scalar; gate on the table.

**Q15 — the 15× swing does not destabilize the optimizer the way described.**
On seq batches only PSR-head params receive gradient (backbone detached, other
heads zero-grad); Adam's per-param moments see *consistent* magnitudes on the
steps where each param updates. The clip (5.0, not 1.0) rarely binds on
non-seq batches. Leave PSR_WEIGHT alone until RF6; if you then un-detach PSR,
*that* is the moment to drop PSR_WEIGHT to ~2–3 (an amplified loss suddenly
touching the backbone is the actual risk).

**Q16 — corrected premise (T=8 already).** Headroom exists (~7 GB); go to
T=16/stride 2 at RF6 only if PSR F1@±3 plateaus < 0.5, and keep seq_every=4
(don't also double the seq-batch count; one variable at a time).

---

## Section 5 — Infrastructure (Q17–Q20)

Q17/Q18 answered in the GPU playbook above.

**Q19 — implemented (F21).** The linear rule (Goyal), not sqrt, is what F4
committed to and it's exact here: peak per-sample intensity =
`factor·5e-4/EFFECTIVE_BATCH`. With batch 4×4=16 the fixed 0.75 was a **1.5×
overshoot** (2.34e-5 vs paper 1.56e-5) — mild, plausibly fine, but now
`ONE_CYCLE_PEAK_FACTOR='auto'` resolves to EFFECTIVE_BATCH/32 (0.5 at 16, 0.75
at 24, 1.0 at 32) so any future batch change self-corrects. F4b re-applies it
on resume. If the current run survives, no action needed until its next
restart picks this up.

**Q20 — short ablations: yes with two constraints.** (a) **Full data, shorter
epochs** (15), never a 25% subset — subsetting guts tail-class coverage for
the 69-way head and changes the task. (b) Compare each single-task metric to
the **multi-task run at the same epoch**, not to RF10 finals. Rankings for
det/pose stabilize by epoch ~10; activity is genuinely at risk of
rank-flipping late (your instinct is right) — so report the activity ablation
with an explicit "epoch-15 snapshot" caveat, or extend just that one run to
25 epochs. Practical plan: the ablation suite runs on the **3060 in parallel**
(single-task fits 12 GB; that card doesn't crash), so ablations cost zero
5060 Ti time and start today.

---

## New fixes on the branch this round

| ID | Change | Why |
|---|---|---|
| F19 | Sentinel logs `lv_pose_EFFECTIVE` (= max(lv_pose, lv_det)) + cap-active flag | Q11 misreading impossible henceforth |
| F20 | `combined_v2` logged next to combined (degrees-normalized pose term); selection unchanged | v1's pose term is raw-MAE-saturated (81% of epoch-2 combined) |
| F21 | `ONE_CYCLE_PEAK_FACTOR='auto'` → EFFECTIVE_BATCH/32, resolved at scheduler build | batch geometry changes self-correct to paper intensity (current 4×4 run was 1.5× hot) |
| — | Regression suite → 22 tests, all passing | F21 pinned |

**Action for you:** merge this branch (it is rebased onto your current main) —
that delivers F17, F18, F19–F21, and the 22-test suite. F18 especially: the
next fresh-start or ablation run should not repeat the ramp² warmup.
