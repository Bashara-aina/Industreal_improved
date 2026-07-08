# 175 — THE ULTIMATE GUIDE: Building the Multi-Task Model that Proves the Hypothesis

**Date:** 2026-07-08
**Scope:** The complete, end-to-end execution blueprint for **Tier F** (multi-GPU / weeks) — one shared-backbone MTL model on IndustReal that is **efficient AND accurate AND beats SOTA on the winnable heads**, with an experimental design that actually *proves* "multi-task helps."
**Consolidates:** 166 (questions) · 172 (audit + corrections) · 173 (proof strategy) · 174 (pinned SOTA/eval) → this is the single document to execute against.
**Rule of the whole guide:** every number that goes in the paper comes from a *measurement under a pinned protocol*, never from prose. If it isn't measured, it doesn't exist.

---

## TABLE OF CONTENTS
0. The proof logic (what "helps" means, falsifiably)
1. The thesis and why it is true on IndustReal (the mechanism)
2. PREFLIGHT — the blocking defects that must die before any run
3. ARCHITECTURE — one shared hierarchical backbone, four heads (fully specified)
4. LOSSES — per head, exact
5. MULTI-TASK OPTIMIZATION — uncertainty weighting + PCGrad (where it plugs into this repo)
6. THE EXPERIMENT MATRIX — the runs that ARE the proof
7. EVALUATION PROTOCOL — pinned, val/test discipline, efficiency
8. SOTA COMPARISON TABLES — the templates to fill
9. THE PAPER — section map, narrative, figures, title
10. RISK REGISTER — what breaks and the graceful fallback
11. EXECUTION TIMELINE — week by week, with go/no-go gates
12. REPRODUCIBILITY & INTEGRITY CHECKLIST
13. APPENDIX — config-flag reference, file/symbol map, evidence index

---

## 0. THE PROOF LOGIC — what "multi-task helps" means, falsifiably

You cannot "show MTL helps" in the abstract. You can only show it against a **matched control that differs in exactly one variable: the presence of the other heads.** Three nested, falsifiable claims (from 173, now the spine of the design):

- **T1 — Efficiency parity (floor, near-certain).** MTL matches single-task per-head accuracy at **N× fewer parameters / one forward pass**. *Falsified if* MTL is >X% worse on any head at equal-or-worse cost. *Proven by* the matched matrix (§6) + measured efficiency (§7.4).
- **T2 — Positive transfer (target, likely on PSR/activity).** MTL **beats** its own single-task baseline on ≥1 head. *Falsified if* every head's Δ ≤ 0. *Proven by* per-head MTL−ST Δ with bootstrap CI, and attributed by leave-one-out (§6).
- **T3 — Beats published SOTA (ambition, winnable on activity + PSR).** MTL beats the published number on ≥1 head **under matched protocol on the test split** (§8). *Falsified if* no head beats SOTA at matched protocol. *Proven by* the §8 tables.

**The honest headline you are driving toward:**
> "A single shared-backbone multi-task model matches or exceeds single-task accuracy on all four IndustReal heads — with positive transfer on procedure-step and activity recognition — at N× fewer parameters and one forward pass, and exceeds published SOTA on activity and step recognition under matched evaluation."

Claim the strongest tier your data supports; the paper must degrade gracefully to T1 (§10).

---

## 1. THE THESIS AND WHY IT IS TRUE HERE

**Thesis (the author's, now evidenced in 172):** Multi-task learning on IndustReal does not fail because of task interference; it failed because of a stack of *implementation/execution* defects that a shared model exposes all at once. Fix the execution and MTL helps.

**Evidence it's an execution problem, not an idea problem (172, receipts):** detection actually reached `det_mAP50_pc = 0.468` in multi-task (the "0.0" was an empty-eval-subsample artifact); activity's training loss was literally `0.0` (zero gradient — a masking/schedule bug, plus a `ramp²` double-ramp bug in `losses.py`); PSR was starved by GELU saturation (now LeakyReLU-repaired) and by a staging curriculum that *zeroed its precision* until epoch 16; V8 activity was fed hash-randomized labels. None of these is task competition.

**Why MTL *should* help on IndustReal (the mechanism — this is the intellectual core):** the four tasks form a natural causal hierarchy plus an attention signal:
- **Detection → PSR is a hierarchy.** Procedure-step recognition is temporal aggregation of per-frame assembly-state detection. A backbone that must localize assembly states is literally building PSR's inputs. → detection is a *strong auxiliary* for PSR (the most likely T2 win).
- **Activity ↔ PSR are cause/effect.** Actions ("plug wheel") cause step transitions; each regularizes the other.
- **Head pose is an egocentric attention prior.** Where the head points ≈ the active workspace; pose supervision biases the backbone toward the manipulated region, aiding detection/activity focus.

This turns "we put four heads on a backbone" into "we exploit the causal structure of assembly understanding." **That mechanism is what a reviewer rewards, and the leave-one-out ablation (§6) is what proves it empirically** (e.g., PSR drops when detection is removed ⇒ the hierarchy is real).

---

## 2. PREFLIGHT — defects that must die before any run (else every result is invalid)

Do not train until all of these are closed. Each is grounded in 172/174 with file:line.

| # | Defect | Where | Fix | Validates |
|---|---|---|---|---|
| P1 | **Hash-randomized activity labels** | `scripts/train_v8_multitask.py:216` (`hash(cls_str)%num_classes`) | Ordered-dict class map from a frozen sorted class list; assert same string→same idx across workers | Activity can be trained at all |
| P2 | **Activity zero-gradient / double-ramp** | `losses.py` activity ramp (F18 site) + label masking path | Assert `loss_act > 0` and non-zero grad at step 0; ramp applied **once** (loss-level only) | Activity actually learns |
| P3 | **Empty-subsample detection eval → 0.0/NaN** | `full_eval_inprocess.py:406` (`det_n_present_classes=0`) | Evaluate on full/GT-stratified val; never let the denominator be 0 | Detection numbers are real, not artifacts |
| P4 | **Staging zeroes PSR/pose precision until epoch 16** | `losses.py` `_get_kendall_stage` (stage1/2 zero `prec_psr`,`prec_hp`) | Set `STAGED_TRAINING=False`, `KENDALL_STAGED_TRAINING=False` (already the config default now); all heads on from epoch 0 | Heads aren't silently frozen |
| P5 | **PSR reported as per-frame 0.7018 (wrong paradigm)** | headline metric | Primary PSR metric = `event_f1`@±3 + τ via `decoder_oracle_bound.py` / `psr_transition_f1.py` | PSR is comparable to STORM/B3 |
| P6 | **0.995 detection cited from wrong file** | `168`/`171` cheat-sheet | Cite D1R checkpoint metadata; use protocol-matched mAP (§8), not native 0.995 | Detection claim survives review |
| P7 | **Val-vs-test split mismatch** | evals run on 5-subject val | Freeze a split config; reserve 10-subject **test** for all SOTA claims (§7.1) | Comparisons are fair |
| P8 | **Fabricated efficiency table (600M/4×)** | `167`/`170` | Delete; measure with `fvcore` (§7.4) | Efficiency claim isn't desk-rejected |

**Preflight exit gate:** a "smoke test" that overfits a 50-clip subset to ~0 loss on **all four heads simultaneously** in one run. If any head can't overfit 50 clips, its plumbing is still broken — do not scale up.

---

## 3. ARCHITECTURE — one shared hierarchical backbone, four heads

The two-backbone V8 (YOLOv8m + MViTv2-S) cannot support the efficiency claim (two backbones ≠ sharing). Tier F uses **one** backbone whose features serve all four heads.

### 3.1 Backbone: Hiera (MAE video-pretrained), weight-shared dual-mode
- **Choice:** **Hiera-B** (base, ~51M) primary; Hiera-L if compute allows. MAE-pretrained on Kinetics. Rationale (174 §5): activity + PSR need temporal; detection needs multiscale; Hiera is the one hierarchical video backbone that gives both. Alternatives if integration stalls: UniFormer-V2 or MViTv2 + added multiscale detection neck. Avoid plain-ViT (single-scale, bad for detection).
- **The resolution tension (name it, don't hide it):** temporal heads want *many low-res frames* (16×224²); detection wants *few high-res frames* (720×1280). Solution that keeps ONE backbone: **weight-shared dual-mode forward** —
  - **Temporal mode:** 16-frame clip @ 224² → stage-4 spatiotemporal features → activity/PSR/pose.
  - **Detection mode:** the clip's key frame(s) @ higher res (e.g., 448–640) through the **same Hiera weights** → multiscale maps {stride 8,16,32} → detection FPN.
  - Weights are shared (so the param/storage efficiency claim holds); the two modes cost different FLOPs (measured honestly in §7.4). This is the crux design decision — it is what lets you be *both* efficient (shared weights) *and* accurate on detection (adequate resolution).
- **Frozen vs trainable:** start with backbone **trainable** (layer-wise LR decay 0.8) — a frozen backbone is why V5 activity failed (frozen ConvNeXt probe ≈ majority). Provide a `--freeze-backbone` variant only for the ablation that measures the freezing cost.

### 3.2 Heads (dims for Hiera-B, feature dim d≈768 at stage-4)
| Head | Input | Module | Output | Params (approx) |
|---|---|---|---|---|
| **Detection** | multiscale {P3,P4,P5} (detection-mode) | FPN neck → decoupled cls+box head, DFL box reg (YOLOv8-style), anchor-free | 24 classes × boxes | ~6–8M |
| **Activity** | stage-4 clip embedding (temporal-pooled), d | LayerNorm → Linear(d, 75) | 75 logits (clip-level) | ~0.06M |
| **PSR** | sequence of per-clip embeddings | small **causal** TransformerEncoder (3 layers, LeakyReLU heads — the `model.py:1604` repair) → 11 per-component transition logits → MonotonicDecoder + precedence | 11 transition logits/frame | ~2–3M |
| **Pose** | stage-4 clip embedding, d | MLP(d→256)→LeakyReLU→Linear(256, 6) → renormalize fwd,up | 6D (fwd3+up3) | ~0.3M |

**Total ≈ 60M** (Hiera-B 51M shared + ~9M heads). Four separate single-task models ≈ 4×(51M+head) ≈ **210M**. → **~3.5× parameter saving** (measure exactly in §7.4 — do not assert until measured).

### 3.3 Input pipeline (pin these; they are part of the protocol)
- Clip: 16 frames, temporal stride 8 (matches the frozen-probe setup, `activity_mvit_probe/results.json`), Kinetics normalization, 224² for temporal mode.
- Detection key frame(s): center frame @ 448–640, IndustReal aspect-ratio-preserving letterbox.
- Class taxonomy **frozen in config**: activity 75-class (`NUM_CLASSES_ACT=75`) for SOTA comparison; optionally also emit the 69-group view via `ACT_ID_TO_GROUP` for the secondary table. **Never** a `hash()` map (P1).

---

## 4. LOSSES — per head, exact

- **Detection:** YOLOv8-style — BCE classification + CIoU box + DFL, over 24 assembly-state classes (`DET_CLASS_NAMES`, verified 1..24 at `config.py:244`). Keep the GT-balanced sampler.
- **Activity:** class-balanced cross-entropy on 75 classes; inverse-frequency or effective-number weights (IndustReal AR is long-tailed). Label smoothing 0.1. Ignore-index for unlabeled frames — but assert not *all* are ignored (P2).
- **PSR:** per-component **BCE with inverse-prevalence weights** (`PSR_COMP_WEIGHTS`, already applied per `SOTA_STATUS` "Q36") on the 11 transition logits, + optional **precedence-consistency** term (reviewer-3 §3: add the precedence matrix as a *training* signal, currently decode-only; expected +0.05–0.10 F1). Primary metric downstream is `event_f1`@±3, so supervise transitions, not just static states.
- **Pose:** cosine/geodesic loss `(1 − cos(fwd_pred,fwd_gt)) + (1 − cos(up_pred,up_gt))` on renormalized vectors (matches the arccos eval, `gt_pose_variance.py:40`). Keep 6D continuous (172 §B6 — do **not** switch to quaternions). Position optional and **unreported** until units verified.

---

## 5. MULTI-TASK OPTIMIZATION — the "correct execution" that makes MTL help

Uncertainty weighting sets *scale*; gradient surgery resolves *conflict*. You need both. Kendall alone cannot rescue a head — it redistributes existing gradient, it cannot create signal (172 §E2).

### 5.1 Loss scalarization: learned uncertainty weighting (keep, with the guards)
Use the existing learned-Kendall path (`losses.py:1691+`), **not** fixed weights:
- `KENDALL_FIXED_WEIGHTS=0`, `KENDALL_HP_PREC_CAP=True` (this cap — `losses.py:1716` — is what stops pose from hijacking the shared backbone; it was a genuine fix, keep it).
- Per-task log_var clamps already present (`act_min=-4`, `psr/pose_max=2`).
- **log_var LR warmup** (172 §D8): put log_vars in their own param group, lr 1e-3, linear warmup over the first ~2 epochs, no weight decay — prevents a head's uncertainty collapsing before it has learned anything.

### 5.2 Gradient conflict resolution: PCGrad (the methods contribution)
Replace the plain `total = Σ prec_i·loss_i; total.backward()` scalarization on the **shared backbone** parameters with PCGrad projection:

```
# per step, for shared-backbone params only:
for each task i:  g_i = ∇_shared (prec_i · loss_i)        # retain_graph across tasks
for each task i:
    g_i^PC = g_i
    for each task j ≠ i (random order):
        if cos(g_i^PC, g_j) < 0:                          # conflicting gradients
            g_i^PC -= (g_i^PC · g_j / ||g_j||²) · g_j     # project onto normal plane
shared.grad = Σ_i g_i^PC                                   # deconflicted update
# head params: normal per-head grads (no sharing → no conflict)
```
- Cost: 4 task-backwards/step (retain_graph) — acceptable for 4 tasks on multi-GPU. If too slow, use CAGrad (one solve) or gradient-vaccine.
- Plug point: this repo scalarizes in `losses.py` and calls backward in `train.py`; add a `MTLBalancer` that intercepts between loss computation and `optimizer.step()`. Keep it behind a flag (`MTL_GRAD_SURGERY=pcgrad|none`) so the **with/without ablation** (§6) directly measures its value — that ablation *is* a paper result ("gradient surgery converts interference from a failure into a managed trade-off").

### 5.3 Schedule, precision, sampling
- **Optimizer:** AdamW. Backbone lr 1e-4 with layer-wise decay 0.8; heads lr 1e-3; cosine schedule; 3-epoch linear warmup (`STAGE3_WARMUP_EPOCHS`-style). Weight decay 0.05 (backbone), 0 (log_vars, norms).
- **AMP:** `MIXED_PRECISION=True`, `AMP_DTYPE='bf16'` (config note `config.py:674` says bf16 is safe and ~1.5–2× throughput; the PSR-spike instability was an fp16-GradScaler issue, absent in bf16). Keep NaN guards on **val** metrics (the epoch-11 bad-checkpoint bug, `SOTA_STATUS` AC-1).
- **No staging:** `STAGED_TRAINING=False`, `KENDALL_STAGED_TRAINING=False`. All heads from epoch 0. Staging is what produced the "activity/PSR dead until epoch 16" artifact (172 §E8, P4).
- **Task-aware sampling:** compose batches so sparse-positive heads see signal every step — PSR-transition-positive frames and rare detection classes must not be starved; class-balanced activity sampler. This is what stops the small-gradient classification heads from being drowned by detection/pose.
- **Grad clip:** global-norm 1.0.

---

## 6. THE EXPERIMENT MATRIX — the runs that ARE the proof

Same backbone, same data, same schedule for every cell; the **only** variable is which heads are present / which mechanism is on. This table is the paper's central result.

| Run | Det | Act | PSR | Pose | Purpose / read-out |
|---|:--:|:--:|:--:|:--:|---|
| ST-Det | ✓ | | | | single-task detection baseline |
| ST-Act | | ✓ | | | single-task activity baseline (the control V5 never had) |
| ST-PSR | | | ✓ | | single-task PSR baseline (transition-F1) |
| ST-Pose | | | | ✓ | single-task pose baseline |
| **MTL-All** | ✓ | ✓ | ✓ | ✓ | the model; per-head Δ vs ST = **T2** |
| MTL-All+PCGrad | ✓ | ✓ | ✓ | ✓ | value of gradient surgery (§5.2 ablation) |
| LOO-noDet | | ✓ | ✓ | ✓ | PSR drop here ⇒ detection is PSR's auxiliary (**mechanism proof**) |
| LOO-noPose | ✓ | ✓ | ✓ | | is pose a useful auxiliary or dead weight? |
| MTL-frozenBB | ✓ | ✓ | ✓ | ✓ | cost of freezing the backbone (explains V5's activity failure) |
| **Backbone-swap** | ✓ | ✓ | ✓ | ✓ | rerun MTL-All with ConvNeXt vs Hiera → interference ranking **moves** ⇒ interference is representation-mediated, not intrinsic (**the single best figure**, 172 §G2) |

**Read-outs:**
- **Transfer (T2):** per-head MTL-All − ST-* Δ, bootstrap CI. Expect PSR ↑ (detection auxiliary), activity ≈/↑, detection ≈, pose ≈.
- **Attribution:** LOO rows name which task helps which → confirms §1 mechanism.
- **Method value:** MTL-All vs MTL-All+PCGrad → gradient-surgery contribution.
- **Representation-mediation:** backbone-swap → interference ranking changes with the backbone.
- **Efficiency (T1):** MTL-All vs Σ(ST-*) params/FLOPs/latency (§7.4).

This is 8–10 runs. On multi-GPU / weeks it is very feasible, and it is the difference between "we believe MTL helps" and "we measured that MTL helps, *why*, and *at what cost*."

---

## 7. EVALUATION PROTOCOL — pinned (from 174), non-negotiable

### 7.1 Split discipline
- IndustReal subject split **12 train / 5 val / 10 test**. Freeze subject IDs in a config file read by every run and every eval. Val (= {05,14,20,24,26}) is for **model selection only**; **test** (10 subjects, confirm IDs from the official release) is for **every headline/SOTA number**. Never mix.

### 7.2 Per-head metrics (exact)
- **Detection:** COCO mAP@0.5 in **both** protocols — annotated-frames (↔ WACV 0.838) and entire-videos (↔ 0.641). `eval_yolov8m.py` / `full_eval_inprocess.py`.
- **Activity:** clip-level top-1/top-5 on **75 classes** (↔ MViTv2 65.25/87.93). Report 69-group only as a secondary "our-task" view; never compare per-frame to their clip number.
- **PSR:** **`event_f1`@±3 + POS + τ** via `decoder_oracle_bound.py` (B3/STORM protocol). POS → appendix with the null-model disclosure (all-zeros POS=0.9995). Per-frame 0.7018 → secondary only.
- **Pose:** forward + up angular MAE (`degrees(arccos(cos))`) with **bootstrap CI** (reuse `bootstrap_ci.json` harness, 1000 resamples, frame-weighted).

### 7.3 Statistical rigor
- Bootstrap 95% CI (1000 resamples, seed 42, frame-weighted) on every headline number.
- Report per-recording spread, not just the mean (the pose per-recording table in `bootstrap_ci.json` is the model).
- For MTL−ST Δ, report the CI of the Δ, not two overlapping CIs.

### 7.4 Efficiency measurement (replaces the fabricated table)
- `fvcore`/`ptflops`: params + FLOPs for MTL-All (temporal mode and detection mode separately) and each ST config.
- Measured FPS (`eff_fps`, already logged) and peak VRAM, identical hardware, batch=1 and batch=N.
- Report **params/storage/one-pass latency** as the efficiency win (real); report FLOPs honestly (two frozen-weight modes may exceed a single ConvNeXt pass — do not spin it).

---

## 8. SOTA COMPARISON TABLES — templates to fill (test split, matched protocol)

**Table A — Accuracy vs SOTA (fill from test-split runs):**
| Head | Ours (MTL-All, test) | SOTA anchor | Protocol match | Verdict |
|---|---|---|---|---|
| Detection mAP@0.5 (annotated / video) | __ / __ | WACV 0.838 / 0.641 | dual-protocol | target parity |
| Activity top-1 (75-cls clip) | __ | MViTv2 **65.25** | 75-cls clip | **beat** target |
| PSR event-F1@±3 / τ | __ / __ | STORM **0.901** / 15.5s (B3 0.883/22.4s) | transition, ±3 | **beat/near** |
| Pose fwd/up MAE (°) | __ / __ | none | — | **first baseline** |

**Table B — MTL vs single-task (the hypothesis, fill from the matrix):**
| Head | ST | MTL-All | Δ (95% CI) | Transfer? |
|---|---|---|---|---|
| Detection | __ | __ | __ | |
| Activity | __ | __ | __ | |
| PSR | __ | __ | __ | |
| Pose | __ | __ | __ | |

**Table C — Efficiency (fill from §7.4):**
| | Σ 4×single-task | MTL-All | Saving |
|---|---|---|---|
| Params (M) | __ | __ | __× |
| FLOPs (G) temporal / det | __ | __ | __ |
| Latency / VRAM | __ | __ | __ |

---

## 9. THE PAPER — section map, narrative, figures, title

- **Narrative (G5, 172):** "Multi-task learning is a **magnifying glass, not a wrecking ball** — a shared model surfaces every latent per-head defect at once; single-task runs hide them. Once execution is fixed, the causal structure of assembly understanding makes MTL *help*."
- **Title options:** *"Multi-Task Learning as a Magnifying Glass: Diagnosing and Fixing Per-Head Failure on IndustReal"* or *"It Wasn't the Multi-Task: Efficient, Accurate Joint Assembly Understanding on IndustReal."* (Avoid "Kendall" in the title — it's a detail, not the hero.)
- **Section → result mapping:**
  1. Intro: the magnifying-glass thesis + the four-way contribution.
  2. Related work: MTL (uncertainty weighting, PCGrad/CAGrad, negative transfer) · egocentric AR · PSR (STORM/B3) · assembly-state detection · ego head-pose (the gap).
  3. Method: shared Hiera + dual-mode + 4 heads (§3) · uncertainty weighting + PCGrad (§5).
  4. **Diagnostics/Failure Analysis (foreground, not Limitations):** the defect taxonomy (§2) — this is a contribution.
  5. Experiments: the matrix (§6), Tables A/B/C.
  6. **Key figure:** backbone-swap interference-ranking (§6) — interference is representation-mediated.
  7. Conclusion + future work (V9 unified pretraining, hierarchical activity from the verb-antonym confusion structure — 172 §B8).
- **Main vs supp:** main = defect taxonomy, backbone-swap, Tables A/B/C, pose first-baseline; supp = threshold sweeps, per-component PSR, confusion matrices, Kalman, POS null-model, D4 decoder transfer, efficiency micro-benchmarks, hash-bug repro.

---

## 10. RISK REGISTER & GRACEFUL FALLBACK

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| No positive transfer on some heads (negative transfer is common) | Medium | **T1 still holds** — report honest per-head Δ (some +, some ≈); "cost of sharing" is credible science, not failure |
| Hiera integration eats the budget | Medium | Tier M proves the hypothesis on the existing backbone; backbone swap is an accuracy upgrade, not a proof requirement |
| Detection res tension hurts mAP | Medium | dual-mode (§3.1); if still low, report parity honestly and lean on PSR/activity for "beats SOTA" |
| Test-split IDs uncertain | Low | pin from official IndustReal release before any test-set number (P7) |
| PCGrad too slow | Low | fall back to CAGrad (single solve) or uncertainty-weighting-only + report the without-surgery result |
| Activity still won't beat 65.25 | Medium | claim "closes X% of the frozen-probe→SOTA gap" + PSR as the beat-SOTA head |
| **Non-negotiable** | — | the matched single-task baselines. Without them there is no proof, full stop. |

---

## 11. EXECUTION TIMELINE — week by week, with go/no-go gates

- **Week 1 — Preflight + integration.** Close P1–P8 (§2). Freeze split config + activity taxonomy. Integrate Hiera dual-mode + 4 heads. **Gate:** smoke test overfits 50 clips to ~0 loss on all 4 heads. *No pass, no scale-up.*
- **Week 2 — Baselines + first MTL.** Run ST-Det, ST-Act, ST-PSR, ST-Pose, MTL-All (no surgery yet). **Gate:** MTL-All is non-degenerate on all 4 heads on val; Table B has real Δs.
- **Week 3 — Method + ablations.** Add PCGrad; run MTL-All+PCGrad, LOO-noDet, LOO-noPose, MTL-frozenBB, Backbone-swap. **Gate:** the mechanism figure (LOO/backbone-swap) tells a coherent story.
- **Week 4 — Test-split eval, efficiency, writing.** Re-eval best runs on the **test** split; `fvcore` efficiency; fill Tables A/B/C; draft the paper against §9. **Gate:** every number in the paper traces to a measured artifact under a pinned protocol.

Parallelize across GPUs: baselines and MTL runs are independent; the eval harness is shared.

---

## 12. REPRODUCIBILITY & INTEGRITY CHECKLIST

- [ ] Split config (subject IDs) committed; every run reads it; val/test never mixed.
- [ ] Activity taxonomy (75-cls) frozen in config; label map is an ordered dict, not `hash()`.
- [ ] Seeds fixed (data, init, bootstrap=42); log the seed per run.
- [ ] Every checkpoint records SHA + its native + protocol-matched metrics (avoid the 0.995-vs-0.0004 provenance trap, 172 C-1).
- [ ] Val-metric NaN guard on checkpoint selection (epoch-11 bug, AC-1).
- [ ] Efficiency numbers from `fvcore`, not estimates; both forward modes reported.
- [ ] PSR: primary = event-F1@±3 + τ; POS in appendix + null disclosure.
- [ ] Every §8 number is test-split, matched-protocol, with bootstrap CI.
- [ ] The §2 defect taxonomy is disclosed (it's a contribution, and it's honest).
- [ ] No fabricated multipliers anywhere; delete 167/170's 4×/600M table.

---

## 13. APPENDIX

### 13.1 Config-flag reference (the knobs that matter, `src/config.py`)
| Flag | Set to | Why |
|---|---|---|
| `KENDALL_FIXED_WEIGHTS` | `0` | learned uncertainty weighting (§5.1) |
| `KENDALL_HP_PREC_CAP` | `True` | pose can't hijack the shared backbone (`losses.py:1716`) |
| `STAGED_TRAINING` | `False` | all heads from epoch 0 (kills the P4 artifact) |
| `KENDALL_STAGED_TRAINING` | `False` | no double curriculum |
| `MIXED_PRECISION` / `AMP_DTYPE` | `True` / `'bf16'` | throughput, bf16-stable (`config.py:674`) |
| `NUM_CLASSES_ACT` | `75` | SOTA-comparable taxonomy (`config.py:275`) |
| `PSR_COMP_WEIGHTS` | inverse-prev | class imbalance on rare transitions |
| `DETACH_PSR_FPN` | `False` | let PSR gradient flow (172 §D4) |
| `MTL_GRAD_SURGERY` (new) | `pcgrad` | gradient conflict resolution (§5.2) |

### 13.2 File / symbol map (where things live)
| Concern | File:symbol |
|---|---|
| Multi-task loss aggregation (Kendall) | `src/training/losses.py:1658+` (`use_kendall` block) |
| HP precision cap | `src/training/losses.py:1716` |
| Staging logic | `src/training/losses.py` `_get_kendall_stage` |
| PSR head (LeakyReLU repair) | `src/models/model.py:1604-1611` |
| PSR transition eval (event_f1@±3) | `src/evaluation/decoder_oracle_bound.py:253`, `psr_transition_f1.py:event_f1` |
| Detection mAP (dual protocol) | `src/evaluation/eval_yolov8m.py:397`, `full_eval_inprocess.py:406` |
| Pose angular MAE | `src/evaluation/…` (`degrees(arccos(cos))`, cf. `gt_pose_variance.py:40`) |
| Bootstrap CI harness | `src/runs/rf_stages/checkpoints/bootstrap_ci.json` (producer) |
| Backbone wrappers | `src/models/video_backbones.py` (MViTv2), add Hiera here |
| V8 hash bug (to fix/retire) | `scripts/train_v8_multitask.py:216` |

### 13.3 Evidence index (verified numbers)
| Fact | Source |
|---|---|
| STORM-PSR IndustReal F1 0.901 / POS 0.812 / τ 15.5s | arXiv:2510.12385 (CVIU 2025), Table 1 — **web-verified** |
| WACV detection 0.838 (annotated) / 0.641 (video) | `industreal-sota-benchmarks.md` (Schoonbeek WACV 2024, Table 3) |
| WACV activity MViTv2 65.25/87.93 (75-cls) | same, Table 2 |
| WACV PSR B1/B2/B3 0.779/0.860/0.883 | same, Table 4 |
| Multi-task detection reached 0.468 (not 0.0) | `full_multi_task_tma_tbank_benchmark/logs/metrics.jsonl` ep62 |
| Activity train loss = 0.0 (zero gradient) | same, `train.activity` |
| Pose fwd 9.14° / up 7.78° + CI | `bootstrap_ci.json` |
| Frozen probe 0.3810 (MViTv2) vs 0.2169 (ConvNeXt) | `activity_mvit_probe/results.json` |
| PSR per-comp 0.7018 (different paradigm) | `bootstrap_ci.json`, `SOTA_STATUS.md` |
| Efficiency real: 46.5M params / 245 GFLOPs | `metrics.jsonl` `eff_*` |

---

*This is the whole plan. The proof is the experiment matrix (§6); the architecture (§3) makes efficiency real; the optimization (§5) makes transfer real; the preflight (§2) makes every number valid; the protocol (§7) makes every comparison fair. Execute §11 in order, fill Tables A/B/C, and the hypothesis — multi-task helps, efficiently and accurately — is either proven with receipts or honestly bounded. Either outcome is a strong paper; the fabricated version was the only losing move, and it's gone.*
