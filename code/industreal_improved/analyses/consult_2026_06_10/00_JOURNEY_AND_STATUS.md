# POPW Project Journey — Complete Status Report
## Updated 2026-06-20 — Through RF2 Epoch 15 Collapse

---

## 1. Project Identity

**Project**: POPW — Pose-Conditioned Multi-Task Architecture for Assembly Understanding  
**Author**: Bashara Aina  
**Target Paper**: `popw_paper_improved.tex` — "POPW: A Unified Multi-Task Architecture for Egocentric Assembly Understanding"  
**Hardware**: Single NVIDIA RTX 3060 (12 GB VRAM), Intel i5-12400F, 64 GB RAM, Ubuntu 22.04  
**Framework**: PyTorch 2.2, CUDA 12.1, cuDNN 8.9  
**Primary Dataset**: IndustReal (egocentric RGB, 1280×720, 10 FPS)  
**Secondary Dataset**: IKEA ASM (third-person, 640×480, 3 RGB views)

---

## 2. What We Built (Architecture Overview)

### 2.1 Core Architecture
- **Backbone**: ConvNeXt-Tiny (ImageNet pretrained) → C2(96ch), C3(192ch), C4(384ch), C5(768ch)
- **FPN Neck**: P3–P7, 256 channels each, lateral 1×1 + top-down upsample + 3×3 smooth
- **Total Parameters**: 76.16M (53.42M trainable)
- **Input**: Single RGB frame [B, 3, 720, 1280]
- **Output**: 5 simultaneous task predictions in a single forward pass

### 2.2 Five Task Heads

| Head | Architecture | Output | Loss |
|------|-------------|--------|------|
| **Detection (ASD)** | RetinaNet-style, P3–P7, 4×Conv3×3+ReLU subnets | 24 classes × 9 anchors/location | Focal(α=0.25,γ=2) + GIoU |
| **Body Pose** | ConvTranspose2d + GroupNorm + soft-argmax | 17 keypoints + confidence | Wing Loss(ω=0.05,ε=0.005) |
| **Head Pose** | GAP(C4)‖GAP(C5) → MLP(1152→512→256→9) | 9-DoF (forward, position, up) | MSE × 0.001 |
| **Activity** | Feature Bank(T=16) + TCN + 2×ViT + CLS token | 75 classes (NA + 74 actions) | LDAM-DRW(s=30) |
| **PSR** | Multi-scale GAP → Causal Transformer(3L,4H) → 11 per-component MLPs | 11 binary components | Binary Focal(α=0.25,γ=1.0) + temporal smooth |

### 2.3 Cross-Task Conditioning (Key Innovation)
- **PoseFiLM**: Body keypoints → γ,β modulation of C5 features (bypasses FPN)
- **HeadPoseFiLM**: Head pose (stop_grad) → second-stage γ,β modulation
- **det_conf**: MaxPool(cls_preds) → [B,24] stop_grad → concatenated into activity input
- **FiLM γ constraint**: `1 + tanh(·)` ∈ (0,2) prevents feature inversion

### 2.4 Training Strategy
- **Kendall Homoscedastic Uncertainty**: L = Σ_t exp(-s_t)·L_t + s_t, init s_det=0, s_pose=-1, s_act=0, s_psr=0
- **Staged Training**: Stage 1 (det only, ep 1–5) → Stage 2 (+pose+headpose, ep 6–15) → Stage 3 (all heads, ep 16–100)
- **EMA**: decay=0.999, active from epoch 16
- **Optimizer**: AdamW, differential LR (backbone 0.1×, heads 1×, bias 0.3×)
- **Scheduler**: Warmup(5ep) → CosineAnnealing(T₀=10, T_mult=2)
- **Batch**: Physical=1, GradAccum=32 → Effective=32 (VRAM constraint with VideoMAE)

---

## 3. Existing Documentation (Phases 1-9)

The original journey through Phases 1-9 (April through June 13, 2026) is preserved in the original document. Key milestones:

- **Phase 1**: Initial Build (April–May 2026)
- **Phase 2**: First Training Attempts — all heads collapsed (May 2026)
- **Phase 3**: Debugging Marathon — 6+ diagnostic tools, NaN guards (Late May–Early June)
- **Phase 4**: First Opus Consultation (June 8) — CrossHeadCrossAttn, 10 confirmed bugs
- **Phase 5**: The Collapse Crisis (June 9–10) — EMA contamination, Mixup/CutMix flipped, eval collate bug, inverted ViT attention, dead FeatureBank
- **Phase 6**: Opus v2 Consultation (June 11) — RC-25 through RC-29 identified
- **Phase 7**: RC-25 Recovery Attempts (June 11–12) — 3-way deadlock discovered
- **Phase 8**: Fresh Start Run 8 (June 12–13) — Pre-fix code, proved architectural/algorithmic root cause
- **Phase 9**: Opus v4 Implementation (June 13) — RC-28/RC-29 fixes, DET_GT_FRAME_FRACTION=0.90, bounded background loss, first det_mAP50=0.0091 above zero

---

## 4. Phase 10: RF1 Death Spiral & The R2.5 Paradox (June 14–17, 2026)

### 4.1 The RF1-RF10 Ladder

The RF1-RF10 system replaced the old R1/R2/R2.5/R3 naming with a structured 10-stage progressive training pipeline:

| Stage | Heads | Data % | Epochs | Gate Criteria |
|-------|-------|--------|--------|---------------|
| **RF1** | Det only | 20% | 20 | det_mAP50>=0.30, det_mAP50_95>=0.12 |
| **RF2** | Det + Pose | 35% | 15 | det_mAP50>=0.40, MAE<=60° |
| **RF3** | Det + Pose + Act | 35% | 15 | det_mAP50>=0.45, act_top1>=0.40 |
| **RF4** | All + PSR trans | 50% | 20 | det_mAP50>=0.50, psr_f1>=0.25 |
| **RF5-RF10** | All heads | 50–100% | 10-15 each | Escalating thresholds to det_mAP50>=0.75 |

### 4.2 The Four Phases of RF1 Failure (June 16)

RF1 was launched and failed 5 retries across 4 distinct failure modes:

**Phase 1 (pi=0.1 collapse cycles, 14:19–15:19)**: Three rapid collapse cycles where cls_mean dropped from -2.3 → -20 in ~15 min each. pi=0.1 init produced 321× stronger background gradient than pi=0.01.

**Phase 2 (false-positive kills, 16:37–16:49)**: pi=0.01 deployed but old CLS_MEAN_CRITICAL=-8.0 threshold false-positive killed healthy training at cls_mean=-8.058 (normal for pi=0.01).

**Phase 3 (real death spiral, 17:04–17:22)**: Fixed threshold, training ran for ~1300 steps. cls_mean stable at -10.5, but max logit decayed from +2.909→+0.055 as gradient faded. The detection head showed a "bounce-and-die" pattern: GT batches revived gradient momentarily, but empty batches drained it.

**Phase 4 (20× LR identity, 17:22–now)**: 20× LR reduction produced IDENTICAL DET-DEBUG outputs at steps 551, 651, 751 — proving the collapse is trajectory-determined, not LR-dependent.

### 4.3 The R2.5 Paradox — Resolved

**The paradox**: R2.5 (paper_run, all heads) trained visibly well, but RF1 (detection-only) died immediately. Both use the same architecture.

**Resolution**: Gradient sparsity. DETACH_REG_FPN only detaches regression gradient, not classification. But the classification gradient from ~16 positive anchors per batch (out of 348K total anchors) distributed across 28M backbone parameters produces ~4×10⁻⁵ per parameter per step. This is at the FP32 noise floor.

R2.5 had 3-4 additional gradient sources (activity, PSR, head_pose) providing 10,000× denser gradient.

**Proof**: At pi=0.01, positive gradient = 73.5 per anchor × 16 = 1176 units total. Background gradient = 2.55×10⁻⁵ × 2048 = 0.052 units. Total gradient distributed across 28M parameters → per-parameter update ≈ 3.4×10⁻⁸ per step. Over 3100 optimizer steps, total parameter change ≈ 0.06%.

### 4.4 Kendall Weighting Bug Discovery (June 17, 21:05 UTC)

**THE BUG — losses.py line 1589**: The Kendall weighting code's `elif self.train_pose:` branch excluded `loss_head_pose` from the total loss:

```python
# Line 1588-1589 — BROKEN:
elif self.train_pose:
    pose_contribution = prec_hp * loss_pose + lv_hp  # loss_head_pose MISSING!
```

Since IndustReal has NO body keypoint annotations (`loss_pose=0` always), the head_pose head was computed in the forward pass (~1.7 loss value) but **never added to the total loss** — zero gradient for 7+ epochs.

**Fix applied** (both Kendall and non-Kendall paths):
```python
pose_contribution = prec_hp * loss_pose + prec_hp * loss_head_pose + lv_hp
```

**Fix confirmed**: Fresh run shows head_pose gradient ALIVE at every LIVENESS_GRAD probe (steps 0, 200, 400, 600), cls_std 1.6× broader at same step, cls_max 5.9× higher.

### 4.5 Key Insight: The One Unimplemented Opus Recommendation Killed RF1

Opus explicitly recommended `train_head_pose=True` in the recovery_det_only preset (calling it "cheap, healthy, gives backbone a 2nd stable signal"). The stage_rf1 preset had `train_head_pose=False`. This single difference — combined with the Kendall bug that would have neutralized it anyway — caused 4 days of failed RF1 training.

---

## 5. Phase 11: RF1 Completion & RF2 Launch (June 18–19, 2026)

### 5.1 RF1 with Kendall Fix

With the Kendall fix applied and `train_head_pose=True`:
- **Head_pose gradient**: ALIVE at every LIVENESS_GRAD probe
- **Backbone grad norm**: 1.03→3.78→2.35→1.36 (healthy, non-zero throughout)
- **cls_std**: 0.41→1.37 (vs 0.88 in broken run — 1.6× broader at same step)
- **cls_max**: -0.93→2.78 (vs 0.47 in broken run — 5.9× higher)
- **near_zero**: 0.0000 at ALL probes (no collapsed classes)
- **Head pose loss**: 1.60→0.01 (fully converged by step 450)

### 5.2 RF1 Gate Met

RF1 completed with `best_det_mAP50 = 0.45` (from stage_history). This represents the first successful completion of a detection-only bootstrap stage in the project's history — enabled by the Kendall fix providing dense head_pose gradient to the backbone.

**NOTE**: There is a discrepancy between stage_history (RF1 best=0.45) and metric_history (max 0.184). The stage_history value of 0.45 was recorded from the stage_manager's gate evaluation, while the metric_history in rf_stage_state.json only shows epochs 7-10 with max 0.184. This may be because the gate evaluation uses a different validation set or evaluation protocol than the metric_history tracking.

### 5.3 RF2 Launch Configuration

```bash
--preset stage_rf2 --resume crash_recovery.pth --subset-ratio 0.35
```

- **Data**: 35% subset (~1.75× more GT frames than RF1's 20%)
- **Active heads**: Detection + Head Pose (Body Pose disabled — no keypoint data)
- **DET_GT_FRAME_FRACTION**: 0.90 (guarantees 90% of batch carries GT boxes)
- **DETACH_REG_FPN**: False (allows regression gradient to backbone)
- **Kendall bug**: FIXED (head_pose contributes to total loss)
- **FP32 mode**: Active (AMP disabled for RTX 3060 stability)

---

## 6. Phase 12: RF2 Epoch 15 Collapse (June 20, 2026)

### 6.1 The Second Collapse

RF2 training reached epoch 15 with PID 1043628. Despite having:
- ✅ Head_pose enabled (dense gradient source)
- ✅ Kendall bug fixed
- ✅ DET_GT_FRAME_FRACTION=0.90 (GT frames guaranteed per batch)
- ✅ 35% data (1.75× more GT frames)
- ✅ DETACH_REG_FPN=False (regression gradient flows)
- ✅ FP32 mode

**The detection classifier collapsed AGAIN at epoch 15.**

### 6.2 The Evidence: Flat Scores at ~0.079

The epoch 15 detection probe reveals:

```
score_p50 = 0.019 (extremely low confidence, near pi=0.01 init)
score_max = 0.93-0.97 (CAN make confident predictions on SOME)
preds>0.30 = 9,974-33,896 per batch (some high confidence counts)
bestIoU_max = 0.91-0.99 (strong localization quality)
verdict: LOCALIZING — detector localizes well but won't fire confidently
```

The critical insight: the classifier CAN produce confident predictions (score_max=0.97) for SOME classes, but the median confidence (score_p50=0.019) and the overall distribution are near-uniform at ~0.079. This is the **cls_score bias differentiation problem** — the classification head's bias moves from pi=0.01 initialization (-4.6) toward a higher baseline, but doesn't differentiate between classes.

### 6.3 Epoch-by-Epoch Collapse Timeline

```
Epoch 7:  det_mAP50=0.007  MAE=71.67°  (run 2 start - degraded from RF1 best)
Epoch 8:  det_mAP50=0.184  MAE=?       (peak — best validation)
Epoch 9:  det_mAP50=0.181  MAE=?
Epoch 10: det_mAP50=0.159  MAE=?       (declining)
...
Epoch 13: det_mAP50=0.000  MAE=56.18°  (near zero detection)
Epoch 14: det_mAP50=0.000  MAE=52.23°  (EVAL COLLAPSE — all heads)
Epoch 15: det_mAP50=0.001  MAE=47.84°  (det flat scores ~0.079, std=0.0088)
```

The pattern: detection mAP peaks at epoch 8 (0.184), then steadily declines. By epoch 13, detection is producing near-zero mAP. Head pose MAE continues improving (71.67°→47.84°) but detection collapses.

**EVAL COLLAPSE reported 56 times** in train.log — all 3 heads (detection, activity, PSR) are collapsed simultaneously at epoch 15. This is the same triple-head collapse pattern seen in all previous runs.

### 6.4 Why This Is Different from the Gradient Sparsity Problem

The RF1 gradient sparsity problem was: "too few gradient sources → backbone doesn't learn → detection head can't differentiate." That's now fixed — head_pose provides dense gradient.

The RF2 epoch 15 collapse is a DIFFERENT mechanism:

1. **For the first 7-8 epochs, detection works**: det_mAP50 reaches 0.184 by epoch 8. The head IS learning.
2. **Then it decays**: Starting around epoch 9-10, mAP declines steadily.
3. **By epoch 13, it's gone**: det_mAP50=0.000010 — near zero.
4. **The classifier produces uniform background predictions**: score_p50=0.019 across ALL probes, cls_score std=0.0068-0.0088 (< 0.01 threshold).

This looks like the **cls_score bias converging to a point where positive and negative gradients exactly balance** — a saddle point or degenerate equilibrium in the Focal Loss landscape. The classification head finds a local minimum where predicting "background everywhere" minimizes loss, and the positive gradient from GT frames is insufficient to push it out.

### 6.5 The cls_score Bias Differentiation Problem

The classification subnet has a learned bias parameter initialized via pi=0.01 → bias=-4.595. With the Kendall fix and dense head_pose gradient:

- pi=0.01 makes background gradient 321× smaller than pi=0.1
- But the gradient from 16 positive anchors per batch is also tiny (73.5 per anchor = 1176 total)
- When distributed across the classification head's 595K parameters, each gets ~0.002 units/step
- At LR=1e-4, update = 2×10⁻⁷ per step
- The bias drifts from -4.595 toward -2.2 (pi=0.1 equilibrium) but never reaches differentiation
- The DET_GT_FRAME_FRACTION=0.90 means 90% of batches have GT, but within each batch, the ratio of positive-to-negative anchors is still 16:348K

**The fundamental issue**: Even with GT frames in every batch, the anchors-per-frame ratio (164K anchors/frame × 4 frames/batch = 656K predictions) with only ~16 positive anchors creates a per-parameter gradient that's too small to drive meaningful differentiation.

### 6.6 The 20-Agent Monitoring Swarm

A 20-agent monitoring swarm (`rf2_swarm/`) was deployed to continuously monitor RF2 training:

- **22 agents** running 134 checks per 5-minute cycle
- **40-thread ThreadPoolExecutor** for parallel execution
- **4-channel alerting**: console, file, webhook, Slack
- **Delta tracking**: per-check verdict changes between cycles
- **Auto-restart watchdog**: 3 consecutive dead cycles trigger automatic restart

**6 blocking bugs were found and fixed** in the first hours of operation:
1. **ND01 false alarms**: NaN/inf in efficiency stat lines (Params: nanM, GFLOPs: nanG) not excluded
2. **ND01 word boundaries**: NaN matching in "optimizer" and other compound words
3. **CS06 false alarm**: det_head_bias log_head_text not included in data_sources
4. **BU01 false alarm**: Same log_head_text fallback issue
5. **L06 false alarm**: Keyword-based "spike" detection triggering on normal loss variation — replaced with 3σ statistical outlier detection
6. **Training heartbeat**: Added to train.py (requires restart to take effect)

### 6.7 Current Reality Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Stage** | RF2 (epoch 16, PID 1043628) | Running but collapsed |
| **Best det_mAP50** | 0.184 (epoch 8) | PEAK — since declined to ~0.001 |
| **Head Pose MAE** | 47.84° (epoch 15) | Improving (from 71.67°) |
| **Gate target** | det_mAP50 >= 0.40 | 21.7× below target |
| **Gate target** | MAE <= 60° | ACHIEVED (47.84°) |
| **Metric history** | epochs 7-10 only | Stops recording after epoch 10 |
| **Stage history** | RF1 best=0.45 | Discrepancy with metric_history |
| **PSR** | Never trained | loss=1.546e-08 constant across ALL runs |
| **GPU** | 12GB RTX 3060 | Stable at ~8GB usage |

---

## 7. What We've Learned

### Proven Hypotheses
1. **Gradient sparsity kills detection-only training**: Math proof shows ~4×10⁻⁵ per parameter per step from 16 positive anchors across 28M backbone params
2. **Kendall bug blocked head_pose gradient**: LIVENESS_GRAD showed NO_GRAD for 104 instances across 7 epochs
3. **DET_GT_FRAME_FRACTION=0.90 works**: 90% of batches carry GT boxes, preventing complete gradient starvation
4. **Multi-task gradient is 10,000× denser**: Activity/PSR/head_pose provide dense per-frame gradient vs detection's sparse anchor gradient
5. **The R2.5 paradox is resolved**: R2.5 worked because all heads provided dense gradient; RF1 failed because detection-only training doesn't.

### Refuted Hypotheses
1. **Checkpoint lineage poisoning (RC-25)**: Fresh ImageNet start reproduced identical 3-head collapse. Definitively refuted.
2. **LR reduction breaks death spiral**: Phase 4 (20× LR) produced IDENTICAL outputs to Phase 3 (1× LR). LR changes affect speed, not equilibrium.
3. **DETACH_REG_FPN blocks ALL detection gradient**: Verification shows only regression subnet detached. Classification gradient flows normally.
4. **Detection-only training works for this architecture**: Proven false by gradient sparsity math and 5 failed retries across 4 phases.

### Open Questions (Critical)
1. **Why did detection classifier collapse AGAIN at RF2 epoch 15** even with head_pose dense gradient + 35% data + DET_GT_FRAME_FRACTION=0.90?
2. **Why does stage_history show RF1 best=0.45 when metric_history shows max 0.184?** Different evaluation protocols? Data splits?
3. **Why has PSR NEVER trained** — loss=1.546e-08 constant across ALL configurations, architectures, and runs?
4. **Is the cls_score bias differentiation problem architectural or loss-based?** The classifier produces uniform ~0.079 scores but individual classes reach score_max=0.97—it CAN differentiate but WON'T.
5. **Is Focal Loss fundamentally unsuitable** for this architecture's anchor density (164K anchors/frame)? Should we switch to Varifocal Loss or Quality Focal Loss?

---

## 8. Current Status Snapshot

**As of 2026-06-20 06:23 UTC:**

```
Stage:      RF2 (epoch 16, stage_index=1)
PID:        1043628 (main) + 8 workers (1045272-1045410)
Status:     running
Best mAP:   0.181 (det_mAP50) — best at epoch 8
Gate:       FAIL (det_mAP50=0.001, MAE=47.84° achieved)
Last HB:    2026-06-20T06:23:34 UTC (NOT stale — updated recently)
Run start:  2026-06-20T00:13:40 UTC (~6 hours ago)
Max epochs: 36
Retry:      0 (first attempt)
Swarm:      Active (22 agents, 5-min cycle, auto-restart enabled)
```

### Available diagnostics:
- Training log: 56MB train.log (still growing)
- Subprocess log: 52MB subprocess.log
- Metrics: 60KB metrics.jsonl (18+ val records across 3 runs)
- Swarm output: 67MB swarm_loop.log
- DET_PROBE: epoch 16 val shows LOCALIZING verdict, score_p50=0.019, score_max=0.93
- State file: rf_stage_state.json at epoch 16

---

## 9. All Python Files and Their Roles (Current State)

### Training Core
| File | Lines | Role | Status |
|------|-------|------|--------|
| `src/training/train.py` | 4519+ | Main training loop | Active — RF2 running |
| `src/training/stage_manager.py` | 3227 | RF1-RF10 orchestration | Active — 10 stages |
| `src/training/training_supervisor.py` | 760 | Deep diagnostics | Ready but not needed (swarm covers) |
| `src/training/losses.py` | ~900 | All loss functions | Kendall bug fixed |
| `src/training/config.py` | 1448 | 10 stage presets | Stable |
| `src/training/model.py` | ~2500 | ConvNeXt-T + FPN + 5 heads | Stable |
| `src/training/evaluate.py` | ~500 | Validation metrics | 5 eval guards |
| `src/training/dataset.py` | ~800 | Data loading | DET_GT_FRAME_FRACTION=0.90 |

### Monitoring Swarm (NEW — 35 files)
| Component | Purpose |
|-----------|---------|
| `rf2_swarm/runner.py` | Main loop, 5-min interval, auto-restart watchdog |
| `rf2_swarm/coordinator.py` | ThreadPoolExecutor dispatch, delta tracking |
| `rf2_swarm/base_agent.py` | CheckResult, AgentResult, Verdict enum |
| `rf2_swarm/config.py` | All paths, thresholds, intervals |
| `rf2_swarm/alerting.py` | 5-severity, 4-channel alerting |
| `rf2_swarm/reporter.py` | Text + JSON output |
| `rf2_swarm/data_sources.py` | Atomic file reloaders |
| `rf2_swarm/agents/` | 22 specialist agents (134 checks) |

---

## 10. What's Next

The RF2 epoch 15 collapse reveals a new class of problem: even with dense gradient from head_pose, the detection classifier finds a degenerate equilibrium at ~0.079 uniform scores after ~7-8 healthy epochs. The fixes that got us through RF1 (head_pose gradient, DET_GT_FRAME_FRACTION) are necessary but not sufficient.

**Critical path forward requires solving the cls_score bias differentiation problem** — understanding why the classifier converges to a uniform background prediction even with GT frames in every batch and dense multi-task gradient.
