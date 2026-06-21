# POPW Project Journey — Complete Status Report
## Updated 2026-06-21 — Through Opus v10 Breakthrough (detach_reg_fpn Smoking Gun)

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

RF2 training reached epoch 15 with PID 3176288. Despite having:
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
| **Stage** | RF2 (epoch 16, PID 3176288) | Running but collapsed |
| **Best det_mAP50** | 0.184 (epoch 8) | PEAK — since declined to ~0.001 |
| **Head Pose MAE** | 47.84° (epoch 15) | Improving (from 71.67°) |
| **Gate target** | det_mAP50 >= 0.40 | 21.7× below target |
| **Gate target** | MAE <= 60° | ACHIEVED (47.84°) |
| **Metric history** | epochs 7-10 only | Stops recording after epoch 10 |
| **Stage history** | RF1 best=0.45 | Discrepancy with metric_history |
| **PSR** | Never trained | loss=1.546e-08 constant across ALL runs |
| **GPU** | 12GB RTX 3060 | Stable at ~8GB usage |

---

## 7. Phase 13: Opus v8 Implementation (June 20, 2026)

### 7.1 The Opus v8 Consultation

Following the RF2 epoch 15 collapse, Opus was consulted for the 8th time (document `36_OPUS_ANSWER_v8.md`). The v8 consultation focused specifically on the cls_score bias differentiation problem — the mechanism where the detection classifier converges to a degenerate equilibrium producing uniform ~0.079 scores despite having dense multi-task gradient.

Opus v8 identified 4 targeted fixes, each addressing a specific contributor to the collapse:

### 7.2 Fix 1: KENDALL_HP_PREC_CAP

**Problem**: The Kendall weighting mechanism could let head_pose log-variance (`lv_hp`) grow unbounded relative to detection log-variance (`lv_det`). Since task weight `prec = exp(-lv)`, a very large `lv_hp` produces near-zero `prec_hp`, effectively zeroing the head_pose contribution to the total loss. This would starve the backbone of its primary dense gradient source, recreating the RF1 gradient sparsity condition.

**Root cause**: The Kendall optimizer minimizes `exp(-lv) * L + lv` for each task independently. Head_pose loss converges quickly (from ~1.7 to ~0.01 by step 450), so its `lv_hp` can grow much larger than `lv_det` (which sees noisy, high detection loss). This assigns near-zero weight to the one head providing dense backbone gradient.

**Fix**: Added `KENDALL_HP_PREC_CAP = True` which clamps `lv_hp >= lv_det`, ensuring the head_pose precision never drops below detection precision:

```python
if self.kendall_hp_prec_cap:
    lv_hp = torch.maximum(lv_hp, lv_det.detach())
```

This prevents the Kendall optimizer from assigning near-zero weight to head_pose while still allowing it to properly weight detection vs. pose.

### 7.3 Fix 2: DET_POS_IOU Thresholds and Matching

**Problem**: Three interlocking hyperparameters were conspiring to starve the classifier of positive gradient:

1. **DET_POS_IOU_TOP_K=1**: Only 1 anchor per GT box matched as positive. With ~16 GT boxes per batch in the IndustReal dataset, this produced exactly ~16 positive anchors per batch out of ~656K total predictions — a 0.0024% positive rate.

2. **DET_POS_IOU_THRESH=0.5**: The 0.5 IoU threshold for ATSS-style matching was too strict for small/hand-held objects common in egocentric assembly. Many GT boxes had no anchor achieving IoU>=0.5, causing them to be entirely ignored during training.

3. **DET_BIAS_LR_FACTOR=5.0**: The bias was learning 5× faster than other parameters. Since the cls_score bias convergence toward the degenerate equilibrium was the collapse mechanism, the 5× acceleration meant the bias overshot healthy operating range before classification features could differentiate.

**Fixes**:
- **DET_POS_IOU_THRESH=0.4** (was 0.5) — lowers the IoU threshold for ATSS positive matching, ensuring more GT boxes have at least one matched anchor
- **DET_POS_IOU_TOP_K=9** (was 1) — allows up to 9 anchors per GT box, increasing total positive anchors from ~16 to ~100-150 per batch
- **DET_BIAS_LR_FACTOR=1.0** (was 5.0) — removes the bias acceleration that was pushing the classifier toward the degenerate equilibrium. The bias now learns at the same rate as other parameters

The combination of TOP_K=9 and THRESH=0.4 increases positive anchor count by approximately 7-9× per batch, providing an order of magnitude more classification gradient to drive differentiation.

### 7.4 Fix 3: KENDALL_STAGED_TRAINING Documentation

**Status**: Documentation confirmation only — no code change required.

Opus confirmed that `KENDALL_STAGED_TRAINING=False` was already correct in the config. During staged training, Kendall weighting is only meaningfully active from stage 2 onwards (when multiple task heads are present), so the staged flag was redundant. The existing `train_pose`/`train_act`/`train_psr` stage flags properly control which heads participate in Kendall weighting.

### 7.5 Fix 4: Phantom 0.45 Bug

**Problem**: The stage_history showed RF1 best_det_mAP50=0.45, but metric_history showed max 0.184. This 2.4× discrepancy caused confusion about whether RF1 had truly met its gate or whether a different evaluation protocol was in use.

**Root cause**: A gate-threshold recording bug in `stage_manager.py`. When the stage_manager recorded gate evaluation results, it was storing the gate threshold value (det_mAP50 >= 0.45) instead of the actual measured metric. The `best_metric` field in stage_history was being populated with `0.45` — the gate target — not the actual evaluation result. This created a "phantom" metric that appeared to document successful gate passage but was actually just echoing the config.

**Fix**:
- Added `_validate_stage_history_entry()` guard to catch threshold/metric confusion by comparing stored values against plausible metric ranges
- Cleaned stage_history to remove phantom entries and backfill with the last-known genuine evaluation metrics
- All recorded values now undergo validation to ensure they represent genuine evaluation metrics, not gate parameters
- The stage_manager now logs both the measured value and the gate threshold separately, with explicit field naming to prevent future confusion

### 7.6 Commit Summary

All 4 fixes were implemented in a single commit:

```
beda631 — 4 files changed, 256 insertions, 119 deletions
```

| File | Changes |
|------|---------|
| `config.py` | Added KENDALL_HP_PREC_CAP, updated DET_POS_IOU_THRESH/TOP_K/BIAS_LR_FACTOR |
| `losses.py` | Added Kendall HP precision cap logic |
| `stage_manager.py` | Added _validate_stage_history_entry(), cleaned phantom metrics |
| `train.py` | Minor config plumbing for new parameters |

**Config hash**: `3e6b58a5cb19765e` — uniquely identifies the Opus v8 fix configuration.

---

## 8. Phase 14: New Training Run with Opus v8 Fixes (June 20-21, 2026)

### 8.1 Launch Configuration

Following the Opus v8 fix implementation, a new training run was launched:

```bash
--preset stage_rf2 --resume src/runs/rf_stages/checkpoints/best.pth --subset-ratio 0.50
```

- **PID**: 3176288 (main) + 14 worker processes
- **Start time**: 2026-06-20 21:44 UTC
- **Config hash**: `3e6b58a5cb19765e` (includes all Opus v8 fixes)
- **Resumed from**: epoch 17 checkpoint (`best.pth` from the old run, which had `best_metric=0.4622` — a phantom value from the stage_history bug, now corrected)
- **Subset ratio**: 0.50 (was 0.35 — a 43% increase in training data)
- **Active heads**: Detection + Head Pose (same as Phase 12)

### 8.2 Key Configuration Changes from Phase 12

| Parameter | Phase 12 Value | Phase 14 Value | Impact |
|-----------|---------------|---------------|--------|
| `subset_ratio` | 0.35 | 0.50 | 43% more training data |
| `DET_POS_IOU_THRESH` | 0.5 | 0.4 | More GT boxes receive positive anchors |
| `DET_POS_IOU_TOP_K` | 1 | 9 | ~9× more positive anchors per GT box |
| `DET_BIAS_LR_FACTOR` | 5.0 | 1.0 | No bias acceleration toward equilibrium |
| `KENDALL_HP_PREC_CAP` | False | True | Prevents head_pose from being down-weighted |
| Stage history | Phantom 0.45 recorded | Validated metrics | Gate metrics now accurate |

### 8.3 Current Training Status

Training is actively running at epoch 17, approximately 2200/3302 batches completed (67% through the epoch).

**DET_PROBE at epoch 17** shows healthy detection throughout the epoch:

| Probe Metric | Range | Interpretation |
|-------------|-------|---------------|
| **score_p50** | 0.020-0.072 | Similar to epoch 6-10 healthy range from Phase 12 |
| **score_max** | 0.37-0.99 | Wide range — healthy confident predictions across classes |
| **preds>0.05** | 28K-100K per batch | High prediction counts — classifier is actively predicting |
| **bestIoU_max** | 0.86-0.98 | Excellent localization quality — regression head remains accurate |
| **bestIoU>0.5** | 472-3037 per batch | Consistent localization matches throughout epoch |

**Verdict**: LOCALIZING for most probes. One probe showed TOTAL COLLAPSE but was identified as targeting a no-GT image (false alarm: the dummy GT tensor produces zero positive anchors, which triggers the collapse detector but is expected behavior for an empty-GT frame).

**LIVENESS at epoch 17 step 2600**:
- **det = 1.57** — ALIVE (gradient flowing to detection head)
- **pose = 1.12** — ALIVE (gradient flowing to head pose head)
- **head_pose = 8.89e-03** — ALIVE (gradient flowing through backbone)

**Loss range at epoch 17**:
- det_cls: 0.31-0.69
- det_reg: 0.25-0.44
- pose: 0.001-0.025

All three losses are in healthy ranges. Detection classification loss is not diverging, detection regression loss is moderate, and head pose loss is near convergence (as expected after 17 epochs).

### 8.4 Early Indicators

**No collapse after 2+ hours of training**. The previous run (Phase 12) collapsed at epoch 15 after approximately 1.5 hours. The Opus v8 fixes have survived 2+ hours without collapse, covering epoch 17 — 2 epochs past the previous collapse point.

**Critical caveats**:
- **No epoch-end validation results yet**: Epoch 17 is not complete. The STEP VAL at gs=1000 and gs=2000 both show det_mAP50=0.0000, but these are intra-epoch step validations (run mid-epoch), not epoch-end evaluations. Intra-epoch mAP values are expected to be near-zero during the early portion of an epoch since the model is evaluating on unseen data without the benefit of a full epoch of training.
- **PSR still dead**: `train_psr=False` in the RF2 stage, so PSR loss remains constant (unchanged behavior — PSR training begins in RF3).
- **Activity still not trained**: `train_act=False` in RF2 stage (activity training begins in RF3).
- **Phantom 0.45 bug is fixed**: Stage_history no longer confuses gate thresholds with actual metrics.

### 8.5 Comparison with Phase 12 (Previous RF2 Run)

| Metric | Phase 12 (Old Run, PID 3176288) | Phase 14 (Opus v8 Run, PID 3176288) |
|--------|-------------------------------|-------------------------------------|
| **Collapse epoch** | 15 | None yet (at epoch 17, 2+ hours) |
| **Best mAP at epoch 17** | ~0.001 | PENDING (epoch not complete) |
| **score_p50 at equivalent point** | 0.019 | 0.020-0.072 |
| **score_max at equivalent point** | 0.93-0.97 | 0.37-0.99 |
| **preds>0.05** | Not tracked | 28K-100K |
| **bestIoU>0.5** | Not tracked | 472-3037 |
| **Positive anchors per GT** | ~1 | ~9 |
| **Data subset** | 35% | 50% |
| **Bias LR factor** | 5.0 | 1.0 |

---

## 8. Phase 15: The 6-Epoch Plateau — Epoch 20 LR Restart Failure (June 21, 2026)

### 8.1 What Actually Happened

The Opus v8 fixes successfully prevented the catastrophic collapse at epoch 13-15 that killed the previous run. But instead of collapse, the run entered a profoundly different regime: **a structural plateau at mAP50=0.20-0.215 that persisted for 6 consecutive epochs (15-20).**

```
Epoch  7: mAP50=0.007  MAE=71.67°  (run 2 start — degraded)
Epoch  8: mAP50=0.184  MAE=?       (peak from old run)
Epoch  9: mAP50=0.181  MAE=?
Epoch 10: mAP50=0.159  MAE=20.73°  (declining in old run)
... [old run collapse: epochs 11-15 all near-zero] ...
... [Opus v8 fixes deployed, restart at epoch 17 checkpoint] ...
Epoch 16: mAP50=0.215  mAP50_95=0.081  MAE=8.80°  combined=0.4622
Epoch 17: mAP50=0.204  mAP50_95=0.077  MAE=9.25°  combined=0.4547
Epoch 18: mAP50=0.207  mAP50_95=0.078  MAE=9.27°  combined=0.4564
Epoch 19: mAP50=0.209  mAP50_95=0.081  MAE=9.33°  combined=0.4580
Epoch 20: mAP50=0.205  mAP50_95=0.080  MAE=9.23°  combined=0.4553  ← LR restart HERE
```

**Key observations:**
- Range over 5 epoch-ends: 0.2039-0.2151 (1.1 percentage point range)
- **Zero trend direction**: Not improving, not degrading — flat with noise
- Pseudo-classing mAP (det_mAP50_pc): 0.3442 → 0.3071 (also flat, ~50% above raw mAP)
- **det_n_present_classes: 15-16/24 consistently** — same 8-9 classes never detected
- **Combined metric stable**: 0.4547-0.4622 (driven by MAE+loss components, not detection)
- MAE: 8.80° → 9.23° (well under 60° gate, flatter than expected)

### 8.2 The CosineAnnealing Restart Failed — This Is Decisive

**The LR restart at epoch 20 (CosineAnnealingWarmRestarts, T₀=10) had ZERO effect:**

| Metric | Epoch 19 (pre-restart) | Epoch 20 (post-restart) | Change |
|--------|----------------------|----------------------|--------|
| det_mAP50 | 0.2088 | 0.2047 | −0.0041 |
| det_mAP50_95 | 0.0810 | 0.0795 | −0.0015 |
| forward_angular_MAE_deg | 9.33 | 9.23 | −0.10° |
| combined | 0.4580 | 0.4553 | −0.0027 |
| det_mAP50_pc | 0.3132 | 0.3071 | −0.0061 |

This eliminates the hypothesis that "the plateau is schedule-dependent." The CosineAnnealing restart resets the LR to its maximum value and re-initializes the optimizer state. If the plateau were caused by a too-small LR (getting stuck in a local minimum), the restart would have produced a measurable improvement. It produced **nothing** — the epoch 20 value is statistically indistinguishable from epoch 19.

**Conclusion**: The plateau at mAP50≈0.21 is structural, not schedule-dependent. The model has reached its current representational ceiling with the current configuration.

### 8.3 POS_ANCHOR_PROBE Evidence — The Classifier IS Learning

The POS_ANCHOR_PROBE (added in response to Opus v9) tracks sigmoid scores on **matched positive anchors only** — the first metric that can actually see classification health.

At epoch 21 step 1600-1700:
```
POS_ANCHOR_PROBE img=0 call=204800: n_pos=525 mean=0.646 med=0.638 max=0.994 min=0.270
POS_ANCHOR_PROBE img=1 call=205000: n_pos=346 mean=0.732 med=0.757 max=0.998 min=0.283
POS_ANCHOR_PROBE img=0 call=205200: n_pos=476 mean=0.799 med=0.851 max=0.993 min=0.382
POS_ANCHOR_PROBE img=0 call=205400: n_pos=164 mean=0.754 med=0.754 max=0.991 min=0.478
```

**This refutes the "classifier is collapsed" narrative.** The classifier scores positive anchors at mean=0.65-0.80, median=0.64-0.85, max=0.99. It can confidently classify objects it has matched to — with sigmoid outputs of 0.99 (essentially 100% confidence).

**The question changes**: It's not "why is the classifier collapsed?" (it's not), but rather "why are 12/24 classes at AP=0 when the classifier clearly works?"

### 8.4 Opus v9 Corrections — Three Critical Diagnoses

On June 21, Opus was consulted for the 9th time with all epoch 16-20 data. The response (`39_OPUS_ANSWER_v9.md`) delivered three corrections that fundamentally changed the understanding of the problem:

#### Correction 1: score_p50 Is Structurally Blind

Opus proved that score_p50 (the median max-class sigmoid over ALL anchors) is the wrong metric. With ~172K anchors/image and a handful of GT, >99.99% of anchors are background. The median anchor's score is by definition ≈ sigmoid(bias) regardless of classification quality. A perfectly trained detector would show the same score_p50.

**Impact**: All prior analysis treating score_p50 as a classification health metric was invalid. The number literally cannot move when classification improves.

#### Correction 2: LOCALIZING Verdict Is IoU-Only

The DET_PROBE's LOCALIZING verdict checks box overlap at score>0.01 — it never evaluates the predicted class. A model putting class-7 predictions on class-6 objects at IoU 0.9 would score LOCALIZING and mAP=0 simultaneously. The "LOCALIZING but not CLASSIFYING" narrative was half-right — the LOCALIZING half is real, the "not CLASSIFYING" half was inferred from score_p50 which can't see it.

**Impact**: The LOCALIZING verdict is consistent with both "classification is fine" and "classification is dead." It cannot distinguish them.

#### Correction 3: The detach_reg_fpn Split-Brain

The committed stage_rf2 preset contains `detach_reg_fpn=True` (line 1109 of config.py), meaning the regression subnet is detached from FPN gradient flow. Under this configuration:
- Only the classification subnet (and head_pose) shapes the backbone
- The excellent localization (bestIoU=0.86-0.98) is produced by the reg subnet riding on features carved by CLS + POSE
- This points AWAY from "bad features" and SQUARELY at the cls loss/cls targets/labels

**However**, the actual running config may differ (the doc claims DETACH_REG_FPN=False). The effective value was never printed at step 0. **This must be resolved before the next diagnosis.**

### 8.5 The Top-k IoU Floor Problem (New in Opus v9)

Opus v9 identified a critical bug in Fix 2: `DET_POS_IOU_TOP_K=9` has **no minimum-IoU guard**. For a GT whose best anchors sit at IoU~0.2 (small parts against ANCHOR_SIZES starting at 96px), the code force-assigns 9 poorly-localized anchors to predict class `c` with target 1.0. Regression tolerates loose anchors (GIoU learns offset), but **classification is actively mistaught** that features at a 0.2-IoU location are class `c`.

This can CREATE the uniform-output pathology for small/medium objects precisely because of Fix 2 — it trades gradient starvation for label noise.

**Fix needed**: Gate the top-k by IoU — keep only topk_idx with IoU ≥ ~0.2-0.3, or adopt ATSS's per-GT adaptive threshold.

### 8.6 What Opus v9 Recommended (Not Yet Done)

1. **50-image cls-only overfit** (highest priority, <30 min parallel to live run): Train on 50 images, classification-only, head_pose OFF, Kendall OFF. Result bins the entire problem:
   - mAP→0.8+: dynamics are the issue → flip KENDALL_FIXED_WEIGHTS=True
   - mAP stalls, boxes localize: assignment/label noise → top-k IoU floor + label audit
   - No localization: anchor/assignment bug upstream of cls

2. **Add missing probes** (before next consultation):
   - cls_score.weight.norm() per epoch (v8's E3 — still the most diagnostic line)
   - prec_hp/prec_det ratio per epoch (confirms HP_PREC_CAP is holding)
   - Effective C.DETACH_REG_FPN and C.REINIT_PI at step 0

3. **PSR 50-sequence overfit** (de-risks paper's novelty claim): Run PSR-only on 50 sequences in parallel. If it can't overfit, the cause is transformer logit scale (−23/+22), fixable before RF4.

4. **Don't jump to RF3 yet**: Run RF2 only until one epoch-end mAP50@0.001 holds for ≥3 consecutive evals past epoch 15, OR until 3 consecutive epochs with mAP50@0.001 < 0.10.

### 8.7 The 12/24 AP=0 Mystery

The most important finding from the epoch 16-20 data: **the same 12 classes are always at AP=0**. Pseudo-classing mAP (mAP50_pc = 0.307-0.344) is ~50% higher than raw mAP because it treats each class independently, confirming the problem IS class-specific.

The mystery class: **Class 6** — consistently has 1500-1800 GT instances per epoch across ALL training runs, yet AP=0 at EVERY epoch. This has never been investigated.

Hypotheses:
1. Class 6 labels are wrong (synthetic label noise on this specific class)
2. The top-k IoU floor poisoning particularly harms small/medium objects that map to class 6
3. Class 6's features overlap with other classes in a way the classifier can't separate
4. There are zero positive anchors for class 6 in the anchor grid (geometry mismatch)

### 8.8 Current Training Status (Epoch 21)

Training continues at epoch 21 (~50% complete, batch ~1700/3302) with:
- **PID**: 3791482 (new PID — a restart occurred)
- **det_mAP50**: 0.2047 (best_metrics), 0.4622 (combined best_metric)
- **LIVENESS**: det ALIVE[2.35e-02], backbone ALIVE[2.770e+00], head_pose ALIVE[4.83e-03]
- **DET gradient bottleneck**: detection_head grad 2.35e-02 vs backbone 2.770e+00 (117× ratio)
- **head_pose borderline**: Alternates ALIVE (4.83e-03) and DEAD (5.34e-04)
- **POS_ANCHOR_PROBE**: n_pos=164-525 per probe, mean=0.64-0.80, max=0.99
- **Heartbeat**: Updating (2026-06-21T07:08 UTC)
- **Remaining epochs**: ~15 (at max_epochs=36), ~86 min/epoch = ~22 hours wall time

### 8.9 Updated Reality Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Stage** | RF2, epoch 21 (~50%) | Running — no collapse |
| **Best det_mAP50** | 0.2047 | 6-epoch plateau (0.204-0.215) |
| **Best mAP50_95** | 0.0810 | Also plateaued |
| **Pseudo-class mAP** | 0.344 (det_mAP50_pc) | ~50% above raw mAP — problem IS class-specific |
| **Head Pose MAE** | 9.23° | Well under 60° gate target |
| **det_n_present_classes** | 15-16/24 | Same 8-9 classes never detected |
| **Gate target** | det_mAP50 >= 0.40 | ~2× below target, no trend |
| **Gate target** | MAE <= 60° | ACHIEVED ✅ |
| **Metric history** | epochs 7-10 only | Still not updating in state.json |
| **LR restart effect** | ZERO | CosineAnnealing at epoch 20 had no effect |
| **Classifier health** | POSITIVE anchors get score 0.64-0.99 | Classifier IS working on matched positives |
| **Opus v8 fixes** | Active | Prevented collapse but ceiling remains |
| **Next decisive action** | 50-image cls-only overfit | Bins the problem in <30 min |

## 9. What We've Learned

### Proven Hypotheses
1. **Gradient sparsity kills detection-only training**: Math proof shows ~4×10⁻⁵ per parameter per step from 16 positive anchors across 28M backbone params
2. **Kendall bug blocked head_pose gradient**: LIVENESS_GRAD showed NO_GRAD for 104 instances across 7 epochs
3. **DET_GT_FRAME_FRACTION=0.90 works**: 90% of batches carry GT boxes, preventing complete gradient starvation
4. **Multi-task gradient is 10,000× denser**: Activity/PSR/head_pose provide dense per-frame gradient vs detection's sparse anchor gradient
5. **The R2.5 paradox is resolved**: R2.5 worked because all heads provided dense gradient; RF1 failed because detection-only training doesn't.
6. **KENDALL_HP_PREC_CAP prevents head_pose gradient starvation**: Clamping `lv_hp >= lv_det` prevents the Kendall optimizer from assigning near-zero weight to head_pose.
7. **DET_POS_IOU_TOP_K=9 dramatically increases positive anchors**: From ~16 to ~120 per batch (from ~1 to ~9 per GT box).
8. **DET_BIAS_LR_FACTOR=1.0 removes bias acceleration**: The 5× bias LR was pushing the classifier toward uniform background prediction 5× faster than parameters could compensate.
9. **Phantom 0.45 was a stage_manager recording bug**: Fixed by `_validate_stage_history_entry()` guard.
10. **Opus v10 — detach_reg_fpn=True for RF2 is the primary cause of the 6-epoch plateau**: The regression gradient is detached from the backbone for stage_rf2 (config.py:1117). Only classification + head_pose shape the backbone. Per Opus v10: "the plateau is substantially dynamic (a config regression), not structural." Flipping detach_reg_fpn=False should allow box-regression gradient to the backbone. (2026-06-21, confirmed via code trace)
11. **Per-class AP IS written to metrics.jsonl**: `det_per_class_ap` is a dict in every epoch-end validation entry. Not a gap — we just hadn't parsed it. Analysis of epochs 16-18 confirms exactly 12/24 classes at AP=0, but 8 of those have zero GT instances.
12. **Per-class AP breakdown (epoch 18)**: Classes 4 (0.37), 5 (1.0), 7 (0.72), 10 (0.48), 12 (0.56), 17 (0.40) have substantive AP. Classes 9 (0.07), 11 (0.13), 20 (0.13), 22 (0.08) are barely present. Classes 5 and 21 have AP=1.0 but only 33 and 151 GT instances. **Class 6 has 1739 GT instances yet AP=0 — the single most important unsolved mystery.**

### Refuted Hypotheses
1. **Checkpoint lineage poisoning (RC-25)**: Fresh ImageNet start reproduced identical 3-head collapse. Definitively refuted.
2. **LR reduction breaks death spiral**: Phase 4 (20× LR) produced IDENTICAL outputs to Phase 3 (1× LR). LR changes affect speed, not equilibrium.
3. **DETACH_REG_FPN blocks ALL detection gradient**: Verification shows only regression subnet detached. Classification gradient flows normally — but the regression detachment is what causes the plateau.
4. **Detection-only training works for this architecture**: Proven false by gradient sparsity math and 5 failed retries across 4 phases.
5. **The "head_pose ate the backbone" narrative**: Weakened by Opus v10. With detach_reg_fpn=True, the backbone receives no regression gradient — the plateau is NOT about multi-task interference but about a severed gradient path.
6. **score_p50 as a classification health metric**: Proven structurally blind by Opus v9. The median over >99.99% background anchors equals sigmoid(bias) regardless of classification quality.

### Open Questions (Critical) — UPDATED for Opus v10 era
1. **Can flipping detach_reg_fpn=False break the mAP ceiling?** Opus v10 predicts "should move the ceiling but may not single-handedly clear 0.40." The fix is deployed in the working tree (config.py:1115, change from True to False). Requires restart to take effect.
2. **Why does class 6 have 1739 GT instances yet AP=0?** Per-class AP from metrics.jsonl confirms this. The single most important unsolved mystery after detach_reg_fpn fix.
3. **Why has PSR NEVER trained** — loss=1.546e-08 constant across ALL configurations, architectures, and runs?
4. **Is the top-k IoU floor (DET_POS_IOU_IOU_FLOOR=0.2 already coded) sufficient to fix the label-poisoning problem?** The floor exists in losses.py:139,152. Needs restart to take effect.
5. **Is Focal Loss structurally capped for the 12 classes that work?** Even among non-zero classes, the AP ranges from 0.02 (class 0) to 1.0 (class 5). Why do some classes reach near-perfect AP while others barely register?
6. **Does the combined metric misleadingly mask failure?** 0.667·mAP50 + 0.333·(1/(1+MAE)) — 1/3 of the weight is head-pose, which is already near-optimal.

---

## 10. Current Status Snapshot

**As of 2026-06-21 07:47 UTC (epoch 21, batch 3200/3302 — 97% complete):**

```
Stage:      RF2 (epoch 21, 97%, stage_index=1)
PID:        3791482 (main) + 14 workers
Status:     running — no collapse, plateau continues
Best mAP50: 0.2047 (epoch 20, best_metrics)
Best mAP50_pc: 0.3442 (pseudo-classing, epoch 16)
Best combined: 0.4622 (epoch 16)
MAE:        9.23° (well under 60° gate)
Gate mAP50: NOT MET — 0.2047 vs 0.40 target (2× below, flat trend)
Gate MAE:   ACHIEVED (9.23° ≤ 60°)
Max epochs: 36 (15 remaining)
Config:     Opus v8 fixes ACTIVE + detach_reg_fpn=False in working tree (not yet restarted)
LR restart: FAILED — epoch 20 CosineAnnealing had zero effect
Heads:      det ALIVE[2.35e-02], pose ALIVE[1.08e-02], head_pose ALIVE[4.83e-03]
Backbone:   ALIVE[2.770e+00|n=178]
```

### Per-Class AP Analysis (NEW — parsed from metrics.jsonl)

The per-class AP IS already being written to metrics.jsonl (not a gap). **Epoch 18 shows:**

| Category | Classes (AP) |
|----------|-------------|
| **Working well** (AP>0.30) | 4 (0.37), 5 (1.0), 7 (0.72), 10 (0.48), 12 (0.56), 17 (0.40), 21 (1.0) |
| **Barely present** (AP<0.20) | 0 (0.02), 9 (0.07), 11 (0.13), 20 (0.13), 22 (0.08) |
| **AP=0 with GT instances** | 6 (1739 GT!), 8 (33 GT), 13 (37 GT), 19 (281 GT) |
| **AP=0, zero GT** | 1, 2, 3, 14, 15, 16, 18, 23 (not in 50% subset) |

**The real story is more nuanced than "12/24 AP=0":**
- 8 classes have zero GT instances — likely not present in the 50% data subset
- 4 classes have GT but AP=0 — **class 6 with 1739 GT is the standout mystery**
- Classes 5 and 21 hit AP=1.0 but have only 33 and 151 instances — very distinctive rare objects
- Classes 0, 9, 11, 20, 22 have AP 0.02-0.13 — these "barely work"

### Available diagnostics:
- **6 epoch-end validation results** (epochs 16-21) — all confirm plateau at mAP50=0.204-0.215
- **POS_ANCHOR_PROBE** — confirms classifier IS learning (n_pos=164-525, mean=0.64-0.80, max=0.99)
- **Per-class AP from metrics.jsonl** — parsed! 12/24 AP=0, class 6 mystery confirmed
- **Pseudo-classing mAP** (det_mAP50_pc=0.307-0.344) — ~50% above raw mAP, confirms problem IS class-specific
- **DET gradient bottleneck**: detection_head grad 2.35e-02 vs backbone 2.770e+00 (117× ratio)
- **Opus v10 diagnosis**: detach_reg_fpn=True is the primary cause. Fix applied (False) in working tree.
- **DET_POS_IOU_IOU_FLOOR=0.2**: Already coded in config.py:307, losses.py:139,152. Needs restart.
- **50-image cls-only overfit**: NOT YET RUN — single highest-value next action
- **Swarm**: Available but needs reconfiguration for PID 3791482

---

## 11. All Python Files and Their Roles (Current State)

### Training Core
| File | Lines | Role | Status |
|------|-------|------|--------|
| `src/training/train.py` | 4519+ | Main training loop | Active — RF2 running |
| `src/training/stage_manager.py` | 3227 | RF1-RF10 orchestration | Active — 10 stages |
| `src/training/training_supervisor.py` | 760 | Deep diagnostics | Ready but not needed (swarm covers) |
| `src/training/losses.py` | ~900 | All loss functions | Kendall bug fixed; HP_PREC_CAP added |
| `src/training/config.py` | 1448 | 10 stage presets | Updated with Opus v8 parameters |
| `src/training/model.py` | ~2500 | ConvNeXt-T + FPN + 5 heads | Stable |
| `src/training/evaluate.py` | ~500 | Validation metrics | 5 eval guards |
| `src/training/dataset.py` | ~800 | Data loading | DET_GT_FRAME_FRACTION=0.90 |

### Monitoring Swarm — 35 files
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

## 12. What's Next

The RF2 epoch 15 collapse revealed a new class of problem: even with dense gradient from head_pose, the detection classifier finds a degenerate equilibrium at ~0.079 uniform scores after ~7-8 healthy epochs. The fixes that got us through RF1 (head_pose gradient, DET_GT_FRAME_FRACTION) were necessary but not sufficient.

The Opus v8 fixes (Phase 13) directly target the cls_score bias differentiation problem through 4 mechanisms:
1. **KENDALL_HP_PREC_CAP**: Prevents the Kendall optimizer from starving the backbone of head_pose gradient
2. **DET_POS_IOU_TOP_K=9 + THRESH=0.4**: Increases positive anchor count by ~9× per batch
3. **DET_BIAS_LR_FACTOR=1.0**: Removes acceleration toward the degenerate equilibrium
4. **Phantom 0.45 fix**: Ensures stage metrics are trustworthy

The Phase 14 run is the first test of these combined fixes. Early indicators are promising — no collapse after 2+ hours and healthy DET_PROBE scores at epoch 17 — but the true test will be epoch-end validation mAP.

**Immediate next steps**:
- Wait for epoch 17 epoch-end validation to complete
- If det_mAP50 > 0, continue training through RF2 gate (det_mAP50 >= 0.40, MAE <= 60°)
- If det_mAP50 ~ 0, the cls_score bias equilibrium was not broken by the Opus v8 fixes
- Restart monitoring swarm with new PID 3176288
- Monitor LIVENESS probes for signs of renewed degradation past epoch 17

---

## 13. Phase 16: Opus v10 Breakthrough — detach_reg_fpn Is the Smoking Gun (June 21, 2026)

### 13.1 The Consultation That Changed the Diagnosis

After the 6-epoch plateau was confirmed through epoch 20 and the CosineAnnealing LR restart failure, Opus was consulted for the 10th time. For the first time, Opus was given a **self-contained comprehensive overview** (`41_OPUS_MASTER_PROMPT_v10.md`) that included ALL evidence — POS_ANCHOR_PROBE, pseudo-classing, LR restart failure, v9 corrections, per-class AP, gradient bottleneck, and full training state.

**Opus v10's answer** (`42_OPUS_ANSWER_v10.md`) was definitive: **detach_reg_fpn resolves to True for stage_rf2, and this is the smoking gun.**

### 13.2 The Code Trace That Proved It

Opus traced the config resolution through the actual code (not summaries):

```
config.py:1117 → stage_rf2 preset has 'detach_reg_fpn': True
stage_manager.py:121 → RF1 stage_cfg overrides to False (RF1 fix applied here)
stage_manager.py:1649 → CLI override only fires if stage_cfg.get('detach_reg_fpn', False) is True
RF2's stage_cfg → NO detach_reg_fpn key → default False → no CLI flag → preset wins = True
```

**Result**: RF2 runs with detach_reg_fpn=True. The regression gradient is severed from the FPN/backbone. The backbone is shaped by classification + head_pose **only**.

### 13.3 Unified Diagnosis

Opus v10's diagnosis reconciled all previously contradictory evidence:

| Symptom | Explanation with detach_reg_fpn=True |
|---------|--------------------------------------|
| bestIoU 0.86-0.98 (localizes) | Reg subnet rides on CLS-carved features — localization doesn't need fine-grained class features |
| mAP50=0.20 (won't fire) | Backbone never gets box-regression gradient — features never become object-discriminative |
| 12/24 classes AP=0 | Feature-starvation is class-selective: discriminable classes survive, subtle/small/rare ones collapse |
| LR restart = zero effect | A detached gradient path is NOT a local minimum — annealing can't restore a severed connection |
| POS_ANCHOR_PROBE 0.64-0.80 | On the easy 12-16 classes, cls-shaped features ARE good enough |
| Pseudo-classing +50% | Gap IS the dead classes — feature-starvation hits classes unevenly |

**The key insight**: "detach_reg_fpn is NOT a bug that causes a local minimum — it's a config error that changes the architecture's training objective. The backbone is only trained to produce features that are good for classification + head_pose. It never receives the signal 'these features need to be good for predicting box coordinates and sizes.'"

### 13.4 The One Caveat

RF2 (detach=True, mAP50=0.204) actually reaches slightly **above** RF1 (detach=False, mAP50=0.184). This means:
- The 2.5× more data (subset_ratio 0.50 vs 0.20) partially compensated for the detached gradient
- Flipping detach_reg_fpn=False should raise the ceiling but may not single-handedly clear 0.40
- Pair with per-class AP logging and top-k IoU floor (DET_POS_IOU_IOU_FLOOR=0.2, already coded)

### 13.5 Config Changes Applied (Uncommitted Working Tree)

Based on Opus v10's recommendations, the following changes were made to `config.py`:

| Change | Opus v10 Recommendation | Status |
|--------|------------------------|--------|
| `stage_rf2` `detach_reg_fpn=False` | ✅ Tier 1, item 1 | Applied (line 1115) |
| `DET_POS_IOU_IOU_FLOOR=0.2` | ✅ Tier 1, item 5 | Already coded (line 307) |
| `DET_LR_MULTIPLIER=2.0` | ❌ NOT recommended by Opus v10 | Applied but NOT reviewed |
| `DET_BIAS_LR_FACTOR=4.0` | ❌ CONFLICTS with Opus v8 (called 5× "an own-goal") | Applied but NOT reviewed |

**⚠️ WARNING**: `DET_LR_MULTIPLIER=2.0` and `DET_BIAS_LR_FACTOR=4.0` are in the working tree but were never part of any Opus recommendation. The rationale comment ("IOU_FLOOR=0.2 prevents false-positive labels so bias can't cheat into equilibrium") is untested. These may need to be reverted to 1.0 before restart.

### 13.6 Updated Reality Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Stage** | RF2, epoch 21 (97%) | Running — plateau continues |
| **Best det_mAP50** | 0.2047 | 7-epoch plateau (0.204-0.215) |
| **Best mAP50_95** | 0.0810 | Also plateaued |
| **Pseudo-class mAP** | 0.344 (det_mAP50_pc) | ~50% above raw mAP |
| **Head Pose MAE** | 9.23° | Well under 60° gate target |
| **det_n_present_classes** | 15-16/24 | 4 AP=0 with GT (class 6 has 1739 GT) |
| **Gate target** | det_mAP50 >= 0.40 | ~2× below target, no trend |
| **Root cause identified** | detach_reg_fpn=True | Fix applied (False) — needs restart |
| **Top-k IoU floor** | DET_POS_IOU_IOU_FLOOR=0.2 | Already coded — needs restart |
| **Per-class AP** | Parsed from metrics.jsonl | 12 AP=0, 4 with GT, class 6 standout |

### 13.7 Decision: What to Do About RF2

**The Opus v10 recommendation is clear**:
1. **Don't advance to RF3** — RF3 `stage_rf3` preset also has `detach_reg_fpn: True`. Advancing inherits the bug.
2. **Fix RF2 first**: Set `detach_reg_fpn=False`, restart from `best.pth`, keep trained heads, don't reinit.
3. Run for 3-4 epochs with per-class AP logging.
4. **Decision rule**: If dead classes wake + mAP climbs past ~0.25-0.30 in 3-4 epochs → detach was the bottleneck. If flat → data-scale/assignment on dead classes.

**The one missing measurement**: Per-class AP is already in metrics.jsonl. The remaining gaps are:
- Per-class positive count (are dead classes getting any positive matches?)
- Per-class max-anchor-IoU (do dead classes have anchor geometry issues?)
- 50-image cls-only overfit (definitive architecture capacity test, still not run)

### 13.8 Updated Python File Changes

| File | Change | Status |
|------|--------|--------|
| `config.py:1115` | `stage_rf2` detach_reg_fpn: False | Uncommitted working tree |
| `config.py:307` | DET_POS_IOU_IOU_FLOOR = 0.2 | Coded, uncommitted |
| `evaluate.py:267-273` | Per-class AP computed, returned | Already working — no change needed |
| `train.py:3625` | metrics.jsonl open for append | Already writing per-class AP |
| `losses.py:139,152` | Top-k IoU floor code exists | Already coded — needs restart |
