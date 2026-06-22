# GUIDE 7 — ANSWERED BY AUDIT (what I verified) vs YOURS TO RUN

*I worked the 200-point checklist against your actual code, configs, docs, and the verified
literature. This guide answers everything resolvable by static analysis — with `file:line`
evidence — and hands you only the items that genuinely need a GPU/dataset, each with the
exact command and what to look for.*

> Legend: ✅ verified-in-code · ⚠️ issue found (fix given) · ✏️ answered as a decision/spec
> (no compute) · 🖥️ requires training/eval (yours) · 📄 requires the dataset locally (yours).
> Evidence paths are under `code/industreal_improved/`.

---

## PART 1 — Seven deep findings from the audit (these change your plan)

### F1. It is **4 benchmarkable heads, not 5**. (resolves a reviewer landmine)
Your own loss code states it: *"there are no real keypoint annotations in IndustReal"*
(`src/training/losses.py:1252-1254`). The 17-keypoint **body pose is synthesized from
detection boxes** (pseudo-keypoints, `src/models/model.py:1935-1964`) and is used **only as
a FiLM conditioning signal**; `loss_pose` is `zero` in practice because `'keypoints'` is
never in `targets` (`losses.py:1255-1263`).
- **Benchmarkable heads:** Assembly-State Detection, Activity, PSR, **Head Pose**.
- **Body/hand pose:** an *architectural conditioning component*, not a benchmark row.
- **Action:** In the paper, present body-pose-FiLM as a mechanism (Contribution #2), and
  report metrics for the 4 real heads. This is honest and removes the "where is your
  body-pose ground truth?" rejection. (Updates GUIDE_3/§2.5 and GUIDE_4.)

### F2. Activity and PSR are **already gradient-isolated** from the shared backbone.
`activity_proj` consumes `c5_mod.detach()` + `p4.detach()` (`model.py:2100-2104`); PSR uses
`DETACH_PSR_FPN` (`model.py:2015-2018, 2068-2070`). So the **backbone is shaped only by
detection (+ head pose)**; activity/PSR are stop-gradient *consumers* of shared features +
cross-task conditioning (`det_conf`→activity, FiLM→C5).
- **Consequence for the claim:** "multi-task synergy" for activity/PSR is **not** joint
  representation learning in the current code — it is shared-feature + conditioning transfer.
  This is a *legitimate, literature-aligned* design (stop-gradient MTL), and it is exactly
  why your naive joint runs collapsed (the gradients you isolated are the ones that fight).
- **Consequence for Ablation A (C4):** the "joint" arm = *remove these detaches*. You likely
  already have log evidence it destabilizes — **that collapse is a publishable result**
  ("joint representation learning across these heads induces negative transfer; stop-gradient
  + conditioning is the stable design," cf. Standley ICML'20, PCGrad NeurIPS'20).
- **Good news:** your architecture is *already ~70% the decoupled design* GUIDE_2 recommends.

### F3. `embedding_cache.py` (Phase B) is **broken against the current model output**.
`cache_embeddings()` reads `outputs['activity_proj']`/`'proj_feat'` and `outputs['pyramid']`
(`embedding_cache.py:472-475`), but `model.forward()` returns **none of those keys** — it
returns `c5_mod, det_conf, act_logits, psr_logits, …` and computes `proj_feat` internally
*without returning it* (`model.py:2106, 2151-2166`). Plus the stray `batch_idx := 1` guard
(`:489`) and an unofficial 80/20 split (`:199-204`). **Phase B will not run until patched.**
Exact patches in PART 3. (This is the only thing blocking your decoupled plan.)

### F4. One forward pass is **confirmed** (C1 solid).
`model.forward()` (`model.py:1849-2166`) runs backbone→FPN→**all heads** in a single call
and returns one dict. The efficiency claim (C2) is true by construction. `count_parameters`
exists (`model.py:2169`) for the params table.

### F5. The honest metric is **wired**, and every benchmark protocol **already exists**.
My earlier fix routes `best.pth`/gates through `det_mAP50_pc` (`train.py` "HONEST METRIC").
`evaluate.py` already computes: present-class mAP (`det_mAP50_pc:1610`), `det_mAP_50_95`,
clip-level activity (`_compute_clip_level_accuracy:729`, `act_clip_accuracy`), top-5
(`:305-311`), activity confusion matrix (`:959`), `psr_f1_at_t`. You do **not** need to write
new eval code for the headline numbers.

### F6. The **detection** confusion matrix does **not** exist yet (only activity's).
The key figure for reframing detection (the 24×24 Hamming-neighbor story, item 79) needs a
~15-line addition to `evaluate.py`. Spec in PART 3.

### F7. The Kendall head-pose bug is **fixed**; FiLM is **correct**.
`head_pose` is in the total loss (`losses.py:1463-1471, 1576-1582`). FiLM γ=1+tanh∈(0,2)
(`model.py:705`); HeadPoseFiLM uses `head_pose.detach()` (stop-grad, `:2092`). Items 132/135 ✅.

---

## PART 2 — The 200 items, answered

### A. Thesis & claim integrity (1–12)
- 1 ✏️ Thesis fixed in GUIDE_1/§0 and GUIDE_6. · 2 ✏️ Contribution = architecture+method+efficiency. · 3 ✅ One forward pass (`model.py:1849-2166`). · 4 ✏️ C1–C5 map in GUIDE_6. · 5 ✏️ "Benchmarkable" defined (same data/split/metric/protocol). · 6 ✏️ Targets pre-registered (GUIDE_3 §3). · 7 ✏️ Fallback = efficiency-Pareto (GUIDE_6). · 8 ✅ No claim needs detection SOTA. · 9 ✏️ Wins = head pose + efficiency. · 10 ✏️+verify: STORM-PSR F1 0.506/POS 0.497 **confirmed** (arXiv 2510.12385); YOLOv8m 0.838 / MViTv2 0.6525 from your docs — 🖥️ confirm in the PDFs before camera-ready. · 11 ✅ Tasks match IndustReal AR/ASD/PSR (WACV'24). · 12 ✏️ Novelty = this task-combo + two-stage FiLM.

### B. Dataset integrity, splits, leakage (13–28)
- 13 ✅ Official CSVs referenced (`config.py:162-164`); 📄 confirm files on box. · 14 📄 **Verify recording-level split** in the loader (critical; can't check without data). · 15 ✅ Errors in val/test (paper-confirmed). · 16 ✅ Counts in config (24/75/11/36); 📄 confirm vs data. · 17 ✅ 24 = 11-bit state codes (`config.py:180-205`). · 18 ✅ `idx = category_id - 1` verified (`industreal_dataset.py:1134-1141`). · 19 📄 Per-class GT counts (approx in `DET_CLASS_ALPHAS`); recompute on box. · 20 📄 Long-tail plot. · 21 ✏️ **Verify val/test `augment=False`** (you fixed an RF1 train/val aug asymmetry; confirm it holds globally). · 22 ✅ Strides set (`TRAIN_FRAME_STRIDE=3`, `EVAL=1`); 📄 confirm clip protocol matches MViTv2. · 23 📄 Checksum. · 24 ✅ GT box rescale correct (`industreal_dataset.py:1116-1133`). · 25 ✅ Head-pose units handled (`HEAD_POSE_POS_SCALE=100`). · 26 📄 Segment boundaries. · 27 ✏️ Test-set discipline (process rule). · 28 📄 Record dataset release version.

### C. Reproducibility & hygiene (29–42)
- 29 ✅ `SEED=42`; 🖥️ run 3 seeds. · 30 ✅ `CUDNN_DETERMINISTIC=True/BENCHMARK=False`. · 31 ⚠️ Add git-commit logging to run dir (not currently logged) — 1-line. · 32 ⚠️ Dump the *resolved* config after `apply_preset` (presets mutate globals); add a JSON dump at startup. · 33 ✏️ Log lib versions. · 34 ✏️ Log cmd+env. · 35 ⚠️ **Add the runtime==committed assertion** (your Run1/Run2 4.0/2.0-vs-1.0/1.0 trap): log every LR/loss hyperparam at startup and assert against config. · 36 ⚠️ `rf_stage_state.json` not being written (doc 45) — fix if you use the orchestrator. · 37 ✏️ One run = one dir = one log. · 38 ✅ `tests/test_checkpointing.py` exists; 🖥️ run it. · 39 ✅ best.pth saves EMA shadow (`train.py:4391-4396`). · 40 ✏️ Add a `reproduce` script. · 41 ✅ Honest metric drives best/gates (my fix). · 42 ✏️ Archive checkpoints off the ephemeral box.

### D. Phase A — backbone + detection (43–60)
- 43 ✅ `recovery_det_only` preset exists. · 44 ✅ `detach_reg_fpn=False` (preset+config). · 45 ✅ `reinit_pi`. · 46 ✅ Focal sane (`losses.py:337-364`). · 47 ✅ GT-frame fraction 0.9 logic (`config.py:1527-1534`). · 48 🖥️ POS_ANCHOR_PROBE (code exists). · 49 ✅ Track `det_mAP50_pc`. · 50 ✅ NaN guards + `GRAD_CLIP_NORM=5.0`. · 51 🖥️ Per-class AP curve. · 52 📄+🖥️ Synthetic pretrain (`pretrain_synthetic.py` exists; needs synth data). · 53 ✏️ Eval on real only. · 54 📄+🖥️ Class-balanced sampling for rare states. · 55 🖥️ Exit on plateau. · 56 🖥️ Save trunk. · 57 ✅ ConvNeXt-T justified (config; [5,30]). · 58 🖥️ Grad-ckpt invariance. · 59 ✅ AMP off (`MIXED_PRECISION=False`). · 60 🖥️ Visualize predictions.

### E. Phase B — cache + temporal heads (61–75)
- 61 ⚠️ Fix `batch_idx` bug (PART 3). · 62 ⚠️ Fix split to official (PART 3). · 63 ⚠️ **Fix output-key mismatch** (PART 3) — the Phase-B blocker. · 64 🖥️ Cache train/val/test. · 65 ✅ Backbone frozen in cache (`embedding_cache.py:442-445`). · 66 🖥️ fp16-vs-fp32 spot check. · 67 ✅ Cache trains only act+PSR (`CacheTrainer:279-301`); pose in Phase A. · 68 🖥️ Cache↔live parity test. · 69 🖥️ Tune `seq_len`. · 70 🖥️ Confirm temporal engaged. · 71 🖥️ Throughput. · 72 ✏️ Frame as curriculum [25,26]. · 73 🖥️ Save head weights. · 74 🖥️ Cache reproducibility. · 75 ✅ One-pass inference holds.

### F. Detection metric & honesty (76–90)
- 76 ✅ `det_mAP50_pc` primary (my fix). · 77 🖥️ Full-test eval. · 78 ✅ `det_mAP_50_95` computed (`evaluate.py:592`). · 79 ⚠️ **Add 24×24 detection confusion matrix** (PART 3) + 🖥️ run. · 80 🖥️ Localization recall (add). · 81 ✅ Thresh/NMS configured (`DET_EVAL_*`). · 82 🖥️ Run `src/diag_per_class_truth.py`. · 83 ✅ No-GT AP=1.0 flagged (`present_class_gt`). · 84 ✅ GIoU decode clamps (`losses.py:374-389`). · 85 ✅ No more OHEM/γ tuning. · 86 ✏️ YOLOv8m caption notes its synth budget. · 87 🖥️ Synth ablation. · 88 🖥️ FPS. · 89 🖥️ Qualitative figs. · 90 🖥️ Lock number.

### G. Activity (91–104)
- 91 ✅ Clip-level eval exists (`_compute_clip_level_accuracy:729`). · 92 ✅ Top-1/Top-5 exist. · 93 ✅ CB-Focal on, `USE_LDAM_DRW=False`. · 94 🖥️ Track `pred_seen`. · 95 ✅ 75-channel head. · 96 ✅ Clip aggregation logic present (`:854-882`); 🖥️ confirm it argmaxes once per segment. · 97 ⚠️+🖥️ VideoMAE is OFF and is a separate stream — re-enabling needs integration + VRAM, not a free switch. · 98 ✅ Activity confusion matrix exists. · 99 🖥️ Temporal-length ablation. · 100 🖥️ Hit target. · 101 ✅ Mixup/CutMix off. · 102 ✅ Activity gradient-isolated (can't dominate backbone, `model.py:2102`). · 103 🖥️ Per-class. · 104 🖥️ Lock.

### H. PSR (105–118)
- 105 ✅ `psr_f1_at_t` exists. · 106 ✅ Transition targets (`losses.py:1359-1377`, `build_transition_targets`). · 107 ✅ Monotonic decoder + order prior (config; `psr_transition.py`). · 108 ⚠️ **det_conf currently feeds activity, NOT PSR** (`model.py:2100`); feeding detection state into PSR (B2's strength) is a *recommended addition*, not current behavior. · 109 🖥️ Unique patterns. · 110 🖥️ Target. · 111 🖥️ Delay metric (add if claiming it). · 112 ✅ Sequence mode engages transformer (`model.py:2025-2031`). · 113 🖥️ Error-robustness. · 114 ✅ `PSR_COMP_WEIGHTS` set. · 115 🖥️ Full test. · 116 🖥️ Timeline fig. · 117 ✏️ Contrast vs B2. · 118 🖥️ Lock.

### I. Head pose + body pose (119–128)
- 119 🖥️ Report head-pose MAE (docs show ~9°). · 120 ✅ 6D rotation (`USE_GEO_HEAD_POSE=True`, `GeometryAwareHeadPose`). · 121 ✅ No head-pose baseline → uncontested. · 122 ✅ Stop-grad on conditioning (`model.py:2092`). · 123 ⚠️ **Body-pose PCK NOT reportable — no GT** (F1). Drop from results; keep as conditioning. · 124 ✅ Wing + soft-argmax τ (train 1.0/eval 0.1). · 125 ✅ Flip pairs set (moot — pseudo). · 126 🖥️ Overlays (optional). · 127 ✅ Pose doesn't degrade det (`loss_pose=0`). · 128 🖥️ Lock head pose only.

### J. MTL integration: FiLM, Kendall, joint (129–146)
- 129 🖥️ Run the 3-arm matrix. · 130 🖥️ Single-task on same backbone. · 131 🖥️ Joint fine-tune (= remove detaches; expect instability = a result, F2). · 132 ✅ Kendall includes head_pose (`losses.py:1576-1582`); `tests/test_loss_kendall.py` exists, 🖥️ run. · 133 ✅ Fixed λ reported (`KENDALL_HP_FIXED_LAMBDA=0.2`). · 134 🖥️ FiLM ablation (flags `use_hand_film`/`use_headpose_film` exist). · 135 ✅ FiLM γ=1+tanh, stop-grad (`model.py:705,2092`). · 136 ✏️ Cite PCGrad/CAGrad [10,28] as remedy/future work. · 137 🖥️ Compute Δ_MTL. · 138 ✅ `count_parameters` exists; 🖥️ run. · 139 🖥️ FiLM-order ablation. · 140 ✅ One-pass (static); 🖥️ profile to confirm. · 141 ✅ Spatial(A)/temporal(B) split documented + **F2 isolation finding**. · 142 🖥️ Task-affinity. · 143 ✅ NaN/zero guards (`losses.py:1494-1517`). · 144 🖥️ Stability curves (you have collapse logs — use them). · 145 ✏️+🖥️ Honest conclusion (synergy/neutral/Pareto). · 146 🖥️ Final numbers.

### K. Evaluation protocol (147–162)
- 147 ✏️ Match each baseline's protocol. · 148 ✅ COCO mAP + present-class + n_present. · 149 ✅ Clip Top-1/5. · 150 ✅ F1@t/POS. · 151 ✅ Head-pose MAE (PCK moot). · 152 🖥️ Test set. · 153 ✏️ Document protocols. · 154 ✅ Combined deprioritized. · 155 ✅ Honest path. · 156 ✅ EMA eval consistent. · 157 ✅ Native 1280×720. · 158 ✅ `USE_TTA=False`. · 159 🖥️ Metric unit test. · 160 🖥️ Determinism re-run. · 161 ✏️ Single-model. · 162 ✏️ Results CSV.

### L. The proving ablations (163–178)
- 163 🖥️ Ablation A (the core). · 164 🖥️ Ablation B FiLM (flags ✅). · 165 🖥️ Ablation C Kendall (`KENDALL_FIXED_WEIGHTS` toggle ✅). · 166 🖥️ Ablation D decoupled-vs-joint. · 167 📄+🖥️ Ablation E synth. · 168 🖥️ Ablation F seq-len. · 169 ✏️ One variable per run. · 170 🖥️ Ablate on val. · 171 🖥️ "Naive joint fails" (use your existing collapse logs). · 172 🖥️ Make A decisive. · 173–175 🖥️ Transfer/efficiency tables. · 176 🖥️ Which task FiLM helps. · 177 🖥️ Significance. · 178 ✏️ One-sentence conclusions.

### M. Statistical rigor (179–186)
- 179 🖥️ ≥3 seeds. · 180 🖥️ Significance test [37]. · 181 ✏️ Report variance sources [38]. · 182 ✏️ Don't claim wins in the noise band. · 183 🖥️ CIs. · 184 ✏️ Small-sample caveat for rare classes. · 185 ✏️ Per-seed appendix. · 186 🖥️ Statistically defensible "competitive."

### N. Efficiency (187–192)
- 187 ✅ `count_parameters` exists; 🖥️ run for the number. · 188 ✅ `scripts/training/efficiency_report.py` exists; 🖥️ run FLOPs/FPS. · 189 ✏️+🖥️ 1-vs-3 passes (1 confirmed in code). · 190 🖥️ Memory footprint. · 191 ✏️ Accessibility (12 GB). · 192 ✏️ Make the table prominent.

### O. Paper, figures, limitations (193–200)
- 193 ✅ **104 placeholders** counted in `popw_paper_improved.tex`; 🖥️ fill from results. · 194 ✏️ Limitations drafted (PART 4). · 195 ✏️+🖥️ 4 figures: architecture (draw now), FiLM (draw now), detection confusion (needs F6 add + run), MTL bar (needs runs). · 196 ✏️ Rebuttals (GUIDE_4 §7). · 197 ✏️ Negative-transfer framing (F2). · 198 ✏️ Related work (PART 4). · 199 ✏️ Release code/configs. · 200 🖥️ Final gate.

**Tally:** ~88 items answered/verified/decided here (✅/✏️/⚠️); ~112 are 🖥️/📄 (need your GPU
or dataset). The blockers are all in PART 3.

---

## PART 3 — Exact patches you must apply before Phase B (then test on your box)

> I did **not** auto-apply these: they touch the model's hot-path output and an untested
> pipeline, and validating them needs a forward pass + data (which I can't run here). They
> are small and ready to paste. Test with one cached batch before a full run.

### Patch 1 — expose `proj_feat` and `p4` in the model output (`model.py`, the return dict ~2151)
```python
        return {
            'cls_preds': cls_preds,
            # ... existing keys ...
            'c5_mod': c5_mod,
            'det_conf': det_conf,
            'proj_feat': proj_feat,          # ADD — what the activity head/cache consumes
            'p4': pyramid['p4'],             # ADD — for the cache's p4_gap
            'act_logits': act_logits,
            # ... rest unchanged ...
        }
```

### Patch 2 — fix the cache key reads (`embedding_cache.py:472-475`)
```python
            proj_feat = outputs['proj_feat']                              # was activity_proj/proj_feat (absent)
            det_conf = outputs['det_conf']
            c5_gap = F.adaptive_avg_pool2d(outputs['c5_mod'], 1).flatten(1)
            p4_gap = F.adaptive_avg_pool2d(outputs['p4'], 1).flatten(1)   # was outputs['pyramid']['p4']
```

### Patch 3 — fix the stray batch guard (`embedding_cache.py:461-490`)
```python
        for bi, batch in enumerate(tqdm(loader, total=max_batches or len(loader))):
            # ... body ...
            if max_batches and bi + 1 >= max_batches:
                break
```

### Patch 4 — use the OFFICIAL split in `CacheDataset` (`embedding_cache.py:196-204`)
Replace the "first 80% of recordings" heuristic with your official train/val/test recording
lists (from `TRAIN_CSV/VAL_CSV/TEST_CSV`) so Phase-B numbers are comparable to baselines.

### Patch 5 — add the detection confusion matrix (item 79) in `evaluate.py`
After per-class AP is computed, accumulate predicted-vs-GT *class* per matched box (IoU≥0.5)
into a `NUM_DET_CLASSES²` matrix and save it (mirror the activity `confusion_matrix` at
`:959`). ~15 lines. This produces the figure that reframes detection as fine-grained state ID.

---

## PART 4 — Corrections to the earlier guides + ready-to-paste paper text

**Corrections (from the audit):**
- GUIDE_3 §2.5 / GUIDE_4: **drop body-pose as a benchmark row** (F1). Report 4 heads.
- GUIDE_2/4: state that activity/PSR are **stop-gradient consumers** (F2); the "joint" arm of
  Ablation A means *removing* those detaches, and its instability is itself a result.
- GUIDE_5: add "apply PART 3 patches" as the first step of Phase B.

**Limitations paragraph (paste-ready, item 194):**
> *POPW is trained on a single 12 GB GPU, which constrains batch size and precludes a
> Kinetics-pretrained video encoder for activity; detection uses real data only, whereas the
> YOLOv8m reference additionally uses ~260k synthetic images [32], so our absolute detection
> mAP is not directly comparable. IndustReal provides no body-keypoint annotations, so body
> pose is used solely as a FiLM conditioning signal rather than a supervised output. To avoid
> the gradient interference we observed in naive joint training, temporal heads (activity,
> PSR) consume stop-gradient backbone features with cross-task conditioning; we report this as
> a deliberate, stable multi-task design rather than full joint representation learning.*

**Related-work positioning (item 198):** multi-task learning [6,8,33,34] and negative
transfer [9,10,28]; conditioning [7]; assembly/egocentric datasets IndustReal [1], STORM-PSR
[2], IKEA-ASM [31], MECCANO; detectors [3,4,32]; video models [15,16]; transfer/foundation
heads [14,24].

---

## PART 5 — Your minimal critical path (only the 🖥️ that prove the idea)

If you do nothing else, these compute steps close C1–C5:
1. 🖥️ **Phase A** → `det_mAP50_pc` on full test (C3-detection).
2. 🖥️ Apply PART 3 patches → **Phase B** → activity (clip Top-1/5) + PSR (F1@t) on full test (C3).
3. 🖥️ **Head-pose** eval (C3, ~9° already) + 🖥️ `count_parameters`/`efficiency_report` (C2).
4. 🖥️ **Ablation A** (single-task vs frozen-MTL vs joint) — proves/qualifies C4.
5. 🖥️ **Ablation B** (FiLM on/off) — proves C5.
6. ✏️ Fill the 104 `.tex` placeholders + limitations (PART 4) + the 4 figures.

Everything else in the 200-list is rigor/polish around these six. When these six are done with
honest metrics and ≥3 seeds on the headline tasks, **C1–C5 are all evidenced and the idea is
proven.**

*References: see GUIDE_6. Live-verified for [1] IndustReal WACV'24, [2] STORM-PSR
(F1 0.506/POS 0.497), [9] Standley, [10] PCGrad.*
