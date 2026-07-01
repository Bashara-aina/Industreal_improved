# AAIML 2027 — Revised Paper Outline & Section-by-Section Plan

**Target:** 8 IEEE pages, training-pathologies-first framing, blockchain/pilot as secondary context.
**Based on:** 20 IEEE reviewer analyses of popw_aaiml2027.tex.

---

## Title & Abstract

**Title:** Three Infrastructure-Level Training Pathologies in Multi-Task Learning: Evidence from an Assembly Verification System

**Abstract (150 words):**
> We characterize three training pathologies in multi-task learning (MTL) that arise from interactions between standard infrastructure components, are distinct from gradient conflict, and are undetectable by standard monitoring. (1) A per-frame class-balanced sampler combined with a recording-keyed feature bank produces non-temporal sequences (98.3% probability across 58 recordings), silently defeating temporal encoders. The same cadence-mismatch pattern independently corrupts the optimizer: a OneCycleLR scheduler configured for per-step stepping but called per-epoch remains in its rising phase for the entire training run. (2) Kendall uncertainty weighting enters a self-reinforcing spiral under extreme label sparsity (46/74 classes <1%), driving the sparse task's learned precision to 0 (prevented by per-task clamping). (3) Per-parameter gradient norms produce cross-tensor ratios dominated by dimensionality artifacts — a survey of 20 MTL repositories finds 70% vulnerable. Beyond the pathology analysis, we summarize 18 infrastructure fixes with estimated impact ranking and validate the complete pipeline in a factory pilot with x402 blockchain micropayments (20 workers, 0% opt-out). Code at [URL].

---

## Section-by-Section Plan

### 1. Introduction (1.0 page)

**Paragraph 1 — Hook:** "We built POPW, a 4+1 task MTL system on a $299 consumer GPU. Training would not converge. After 10 days of forensic analysis across 6 training runs, we traced the failures not to gradient conflict (the usual suspect) but to three classes of infrastructure bug, each invisible to standard monitoring: (a) two independently-occurring component interface mismatches, (b) a loss-weighting feedback loop under sparsity, and (c) a systematic measurement artifact in gradient probes."

**Paragraph 2 — Contributions (reordered):**
1. **Three training pathologies** with mechanism, mathematical derivation, detection method, and fix.
2. **Community survey**: 70% of 20 MTL repositories vulnerable to gradient measurement artifacts.
3. **Infrastructure lessons from 18 verified fixes** with impact ranking identifying the 4 highest-impact changes.
4. **(Supporting) Factory deployment** with x402 micropayments and IEEE 7005-2021 governance, validated in a 20-worker pilot (summarized; full details in supplementary).

### 2. Related Work (0.8 pages)

**Subsections:**
- **2.1 MTL Training Dynamics** (dominant, 0.5p): Gradient conflict (PCGrad, CAGrad, Nash-MTL). Task suppression. Saddle points. Gap: "All prior work treats infrastructure components — samplers, feature banks, schedulers, probes — as black boxes that function correctly." Position relative to EgoPack: "EgoPack demonstrates MTL on egocentric video; we additionally document how infrastructure failures can prevent MTL from converging."
- **2.2 Assembly Understanding** (0.2p, trimmed): IndustReal, IKEA ASM, Assembly101, STORM-PSR.
- **2.3 Blockchain & Ethics** (0.1p, moved to supplementary reference): "The x402 protocol enables HTTP 402 micropayments; full ethical framework in supplementary."

### 3. System Architecture (1.0 page — trimmed, body pose de-emphasized)

**3.1 Backbone and Shared Features:** ConvNeXt-Tiny + FPN (P3-P7, 256ch). 46M params (post-simple-head), 85 GFLOPs, 4.8 FPS on RTX 3060. Table 1 (params/GFLOPs/latency — keep as-is).

**3.2 Task Heads:** Four heads described in one paragraph each:
- Detection: RetinaNet, 24 classes, Focal + GIoU (5.3M params)
- Head pose: GAP(C4||C5) → MLP(1152→512→256→9). 0.8M params. **Reported**: forward-gaze only (up-vector ~95° is unlearned — disclosed in limitations).
- Activity: Per-frame MLP (150K params). **47 hybrid groups** (NOT 74 classes — classes with >=100 frames standalone, remainder verb-grouped).
- PSR: Causal Transformer, 11 binary classifiers (3.1M params). **Per-frame component recognition** (NOT transition detection — causal mask is a no-op under shuffled sampling).
- Body pose: *Not a claimed task.* Include a 1-sentence note: "A ConvTranspose2d decoder producing 17 COCO-style keypoints was included for FiLM conditioning; these are pseudo-keypoints generated from detection boxes and are not independently evaluated."

**3.3 Two-Stage FiLM:** PoseFiLM modulates C5; HeadPoseFiLM with stop-gradient isolation. Detection confidence feeds activity head (24-dim max-sigmoid concatenated with GAP(C5_mod) and GAP(P4), projected to 512).

**3.4 Training Protocol (NEW — condensed):**
- AdamW (GRAD_CLIP=5.0, WD=1e-3). OneCycleLR with pct_start=0.1, steps_per_epoch=1.
- 3-layer sampling: balanced (count_floor=15) → task-aware boost (det 2x, PSR 1.5x) → DET_GT_FRAME_FRACTION=0.40.
- Kendall uncertainty weighting with per-task bounds (§4.2).
- Full hyperparameter table referenced (config.py in repository).

### 4. Three Training Pathologies (2.5 pages — the core)

**4.1 Pathology 1: Infrastructure Component Interface Mismatch (1.0 page, broadened)**

*General statement:* MTL composes independently-designed components. When these components operate on different cadences — frames vs. sequences, steps vs. epochs — they silently produce invalid behavior. We document two independent instantiations.

*Case Study A — Data Pipeline (existing, corrected):*
- Mechanism: WeightedRandomSampler (per-frame) + FeatureBank (recording-keyed). P(sequential) ≈ 1/58 = 1.7%. 98.3% of bank sequences are from multiple recordings.
- Equation: P(r_i_t = r_i_{t+1}) = sum_r (f_r / total)² ≈ 1/R for R=58. Corrected from v1 where R=12 was used.
- Detection: Prediction entropy <0.1 nats for >90% of frames. Gradient norms remain healthy (>0.01).
- Fix: Replace temporal encoder (8.2M params TCN+2xViT) with per-frame MLP (150K params), eliminating the temporal component that the sampling defeats. (Root-cause note: a true fix would use sequence-level sampling.)

*Case Study B — Optimization Pipeline (new):*
- Mechanism: OneCycleLR configured with steps_per_epoch ≈ 800, but scheduler.step() called once per epoch. total_steps computed as 100×800 = 80,000. Rising phase = 8,000 steps. Only ~100 calls occurred. LR stayed in rising phase for 100 epochs.
- Detection: No error raised. LR increased (slowly). Metrics improved. Discovered only by line-by-line code audit.
- Fix: steps_per_epoch=1, matching epoch-level cadence. SequentialLR [LinearLR(warmup, 2ep) → OneCycleLR(pct_start=0.1, steps_per_epoch=1)] produces: warmup epochs 0-1, rising epochs 2-9 to 5e-4, cosine decay epochs 10-99.

*Generalization:* Table comparing both case studies — same pattern (cadence mismatch), different domain (data vs optimization), same invisibility (no error, degraded training).

**4.2 Pathology 2: Loss Scale Suppression Under Label Sparsity (0.7 pages, corrected)**

*Mechanism:* Kendall weighting L = sum_t e^{-s_t}L_t + s_t. Gradient: dL/ds_t = -e^{-s_t}L_t + 1. Fixed point: s_t* = log(L_t). Under sparsity (46/74 classes <1% in the original 74-class labeling), the activity loss is dominated by head-class predictions under the legacy CB sampler, driving s_act toward large negative values.

*Contributing factor:* DET_GT_FRAME_FRACTION=0.90 originally confined 90% of batch mass to detection-GT frames, giving activity ~0.14 frames/class/batch, artificially depressing L_act.

*Fix:* (1) Balanced sampler replaces CB weighting — every class appears equally, maintaining L_act ~ ln(N). (2) Per-task Kendall bounds: activity min -0.5 (precision max 1.65x), PSR clamped at 0.0 (cannot be suppressed), pose max 3.0 (can be suppressed to 0.05x). These prevent any single task from entering the spiral.

*Note:* This pathology was preemptively fixed before it caused measurable degradation. We characterize it as a theoretical concern that standard Kendall defaults are insufficient under the extreme sparsity common in assembly tasks.

**4.3 Pathology 3: Gradient Measurement Artifacts (0.5 pages, shortened)**

*Mechanism:* Logging ||param.grad|| for individual parameters. Comparing ||W_proj|| (512×1048 = 537,696 elements) to a 1-element bias produces sqrt(537696) ≈ 733x ratio from dimensionality alone.
*Correct metric:* RMS gradient = sqrt(mean(||θ||²)) = sqrt(sum(||θ||²)/d). This removes the dimensionality artifact.
*Prevalence:* Survey of 20 open-source MTL repositories (GitHub, Python/PyTorch, stars >100). Methodology in supplementary. 14/20 (70%) log per-parameter param.grad.norm() without head-level aggregation.
*Impact:* Misattribution of gradient magnitude across heads led to 10 days of hyperparameter optimization targeting a non-existent gap. The correct head-level aggregate shows all heads within 3x.
*Framing:* This is a diagnostic lesson for the community, not a training pathology. It is included because its downstream effects (incorrect hyperparameter tuning) degrade training.

**4.4 Infrastructure Lessons Learned (0.3 pages, new)**

Summary table of 18 verified fixes across 5 categories (optimizer, scheduler, sampler, architecture, evaluation). Four tiers by impact:
- **Tier 1** (~80% of improvement): Simple MLP, OneCycleLR correction, GRAD_CLIP 1.0→5.0, WD 5e-2→1e-3
- **Tier 2** (significant): DET_GT_FRACTION 0.9→0.4, bias/norm WD=0
- **Tier 3** (defensive): PSR warmup, segment-label remap, sampling diagnostic
- **Tier 4** (minor): 10+ additional fixes

Key insight: "Of 18 fixes, 4 account for an estimated 80%+ of total improvement — but each of the remaining 14 was necessary for correctness."

### 5. Empirical Results (1.0 page — real numbers only)

**5.1 Protocol:** IndustReal, 70/15/15 split, 3 seeds (42, 73, 128 — verified against code), bootstrap 95% CI.

**5.2 Primary Results (Table 2):**

| Task | Metric | Value |
|------|--------|-------|
| Detection | Present-class mAP50 | [actual] |
| Activity (47-group) | Clip-level top-1 | [actual] |
| Activity (47-group) | Macro-F1 | [actual] |
| PSR (per-frame) | Component binary acc | [actual] |
| Head pose | Forward-gaze MAE | [actual] deg |

Note: Activity is 47 hybrid groups, not 74 classes. The 74-class collapse is the motivation for grouping and is documented in §4. PSR is per-frame component recognition, not transition detection.

**5.3 Controlled Ablation — Equal-Gradient-Update MTL:**
Both arms receive identical detection gradient updates (achieved via [specify mechanism]). Only difference: multi-task arm adds pose/activity/PSR losses. Result: delta = [actual] mAP50_pc. Interpretation: [actual]% structural interference cost, [actual]% task synergy benefit.

**5.4 FiLM Ablation:**
Full FiLM vs No FiLM vs PoseFiLM only vs HeadPoseFiLM only. Report per-task breakdown with Cohen's d or bootstrap CI.

**5.5 Sequential Single-Task Baseline:**
Measured (not estimated) latency of YOLOv8m + [other models] sequentially on same RTX 3060. Compare to POPW 226ms for 5 tasks.

### 6. Deployment and Factory Pilot (0.5 pages — trimmed)

**One paragraph on deployment:** "PSR step completions trigger x402 micropayments on Solana (devnet latency 537ms, gas $0.0002-$0.001/tx). Workers view a real-time earnings dashboard. Edge-local processing (no worker imagery leaves the factory). IEEE 7005-2021 principles (opt-out, transparency, accountability, data governance)." Full blockchain architecture in supplementary.

**One paragraph on pilot:** "20 workers (12F/8M, age 22-58) at a Tokyo dimsum facility, 2 weeks. Zero opted out. SUS=72.3 (above benchmark 68). NASA-TLX pre→post: 65.2→58.4 (d=0.51, p=0.04 nominal, not significant after Bonferroni). Trust in Automation=4.8/7. Three workers aged 45+ required onboarding (mean 30 min). Detection error analysis: 70% of errors are 1-bit-Hamming-adjacent (visually similar states); remaining 30% split between visually similar state pairs (~18%) and occlusion (~12%)." Full thematic analysis in supplementary.

### 7. Limitations (0.5 pages — new)

From reviewer-provided text covering:
1. Activity on 47 groups, not 74 classes
2. Body pose has no real annotations (pseudo-keypoints from detection)
3. Head pose position MAE unit unverified
4. PSR is per-frame, not temporal transition detection
5. Frozen backbone, unmeasured domain shift
6. Single assembly task (dimsum), generalizability unknown
7. Blockchain oracle problem (employer controls GPU)
8. Pilot N=20 underpowered
9. Survey is convenience sample
10. Single GPU architecture (RTX 3060)

### 8. Conclusion & Future Work (0.3 pages)

- Summary of 3 pathologies with cross-domain applicability
- 18 fixes catalog
- Deployment feasibility demonstrated
- 3 recommendations: (1) scheduler cadence must match calling cadence; (2) gradient probes must aggregate at head level; (3) per-class effective sampling mass must be logged
- Code and model weights at [URL]

### References (~25 entries)

Must add: FPN (Lin CVPR 2017), EgoPack (Peirone CVPR 2024), Cui et al. (CVPR 2019), ConsMTL (CVPR 2025), IEEE 7005-2021 standard.
