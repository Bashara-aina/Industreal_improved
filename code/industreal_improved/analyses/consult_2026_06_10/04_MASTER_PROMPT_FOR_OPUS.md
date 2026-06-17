# MASTER PROMPT FOR OPUS — End-to-End Redesign Request
## POPW Opus Consultation v2 (2026-06-11)

---

## YOUR MISSION

You are consulting on a multi-task video understanding project called **POPW**. The project has built a unified model for 5 tasks (detection, activity recognition, body pose, head pose, procedure step recognition) on the IndustReal assembly dataset. **The model has NEVER produced real, trustworthy multi-task metrics.** Every training run has been corrupted by a cascade of bugs (24 root causes identified).

**Your job**: Analyze the entire codebase (10 Python files, 3 log files, 1 evidence JSON, 4 context MD files) and produce **at least 5 detailed implementation guide MD files** that we can directly implement. These guides must be advanced, detailed, and produce a model that ACTUALLY LEARNS — not one that catastrophically fails.

**We are open to ANY change**: different backbone, different heads, different training flow, different losses, different data pipeline. The only constraints are:
1. One unified model, one forward pass, multiple tasks
2. Must run on a single RTX 3060 (12 GB VRAM)
3. Must target the IndustReal dataset benchmarks
4. Must demonstrate genuine multi-task learning (not catastrophic forgetting)

---

## FILES PROVIDED

### Context Documents (read these first)
| File | Purpose |
|------|---------|
| `00_JOURNEY_AND_STATUS.md` | Complete project history, what works, what doesn't |
| `01_PROBLEMS_ROOT_CAUSES.md` | All 24 root causes with evidence and fix proposals |
| `02_GOALS_AND_BENCHMARKS.md` | Target metrics, baselines to beat, priority ordering |
| `03_ARCHITECTURE_DEEP_DIVE.md` | Module-by-module analysis, data flow, loss architecture |
| `previous_opus_answer.md` | Previous Opus analysis (forensic audit of the collapse) |

### Code Files (in `code/` directory)
| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 2320 | Full POPW model: backbone, FPN, all 5 heads, FiLM, FeatureBank, ViT |
| `losses.py` | 1255 | All loss functions: Focal, GIoU, Wing, LDAM-DRW, Binary Focal, Kendall |
| `config.py` | 824 | Complete configuration with Tier 1-3 flags |
| `train.py` | 4017 | Training loop: staged training, EMA, optimizer, checkpointing |
| `evaluate.py` | 4011 | Evaluation pipeline: mAP, Top-1/5, PSR F1/POS, efficiency |
| `industreal_dataset.py` | 1408 | Dataset loader: all modalities, frame cache, augmentation |
| `head_pose_geo.py` | 251 | Geometry-aware head pose (6D rotation, geodesic loss) |
| `psr_transition.py` | 301 | PSR transition predictor (event-based, monotonic decoder) |
| `roi_detector.py` | 379 | ROI-centric detector (class-agnostic localizer + state classifier) |
| `video_stream.py` | 361 | K400-pretrained video stream for activity |

### Log Files (in `logs/` directory)
| File | Purpose |
|------|---------|
| `train_main.log` | Main training log showing the collapse (tail of full run) |
| `eval_post_retrain.log` | Post-retrain evaluation showing corrupted metrics |
| `reinit_runner.log` | The reinitialization process log |

### Evidence (in `evidence/` directory)
| File | Purpose |
|------|---------|
| `eval_metrics.json` | Full evaluation metrics from the post-retrain eval |

---

## WHAT WE NEED YOU TO PRODUCE

### Deliverable: At Least 5 Implementation Guide MD Files

Each guide should be **self-contained, detailed, and directly implementable**. Not high-level advice — concrete code changes, training configurations, and step-by-step procedures.

### Required Guide Topics

#### Guide 1: "Fix The Measurement Chain"
The zero-GPU-cost experiments that tell us the truth about our model.
- Exact code patches for RC-13 through RC-19 (with line numbers and before/after code)
- The zero-cost `latest.pth` evaluation procedure
- An invariant test suite that catches these classes of bugs permanently
- How to verify each fix worked before spending GPU hours

#### Guide 2: "Architecture Redesign for Multi-Task Learning"
How to restructure the model so all heads can learn simultaneously.
- Should we keep ConvNeXt-Tiny or switch backbone? (Analyze tradeoffs for multi-task)
- Should we keep RetinaNet detection or go ROI-centric? (Concrete architecture for each option)
- How to make the FeatureBank actually work (or replace it with something better)
- How to fix the activity head for clip-level evaluation
- How to redesign PSR for transition prediction
- The two-stage embedding cache pipeline (detailed implementation)
- Parameter budget allocation across heads

#### Guide 3: "Loss Functions and Training Strategy"
How to make the model actually learn all tasks without catastrophic interference.
- The optimal loss recipe: which losses, which weights, which scheduling
- Kendall uncertainty weighting: keep, modify, or replace?
- Staged vs joint training: which is better for this problem?
- How to handle extreme class imbalance (75 activity classes, 11 PSR components)
- Learning rate strategy for multi-task (per-head LR, backbone LR, warmup)
- Gradient management: how to prevent one head's gradients from destroying another
- EMA: when to enable, what decay, how to prevent contamination
- Mixup/CutMix: how to implement correctly (mix inputs, not logits)

#### Guide 4: "Beating Each Benchmark"
Task-specific strategies to hit the target metrics.
- **Detection**: Synthetic pretraining → real fine-tuning pipeline. Anchor calibration. ROI-centric vs RetinaNet tradeoff analysis.
- **Activity**: K400 video encoder fine-tuning. Clip-level training protocol. Cross-task conditioning that actually helps.
- **PSR**: Transition-based prediction. Monotonic decoding. Procedure-order priors.
- **Head Pose**: 6D rotation representation. Geodesic loss. How to get <25° MAE.
- **Efficiency**: Quantify the multi-task advantage. Streaming inference design.

#### Guide 5: "The Complete Training Recipe"
An end-to-end training configuration that produces a learning model.
- Exact config.py settings (with reasoning for each choice)
- The training schedule: epochs, stages, milestones
- The validation procedure: correct eval that measures what training sees
- Kill criteria: when to abort a run
- The minimum viable experiment: smallest experiment that proves the architecture can learn
- The full production run: 100-epoch configuration
- Expected metric trajectories at each stage

### Optional Additional Guides (if you have insights)

#### Guide 6: "Cross-Task Conditioning That Actually Works"
- FiLM conditioning: when does it help, when does it hurt?
- Cross-head attention: is it worth the complexity?
- Knowledge distillation from specialist models
- How to prove cross-task benefit in an ablation table

#### Guide 7: "Data Pipeline Optimization"
- Task-aware sampling: per-task dataloaders
- Synthetic data integration
- Embedding cache design
- How to handle the extreme label sparsity

---

## CONSTRAINTS AND REQUIREMENTS

### Hard Constraints
1. **Single RTX 3060, 12 GB VRAM** — no multi-GPU, no A100
2. **PyTorch 2.x** — existing codebase
3. **IndustReal dataset** — primary benchmark
4. **One model, one forward pass** — unified architecture
5. **Maximum 80M parameters** — must fit in VRAM

### Quality Requirements for Your Guides
1. **Concrete code** — not pseudocode, actual PyTorch code we can paste
2. **Line-number references** — reference our existing code where changes are needed
3. **Before/after comparisons** — show what changes and why
4. **Risk assessment** — what could go wrong with each change
5. **Verification steps** — how to test each change before committing GPU hours
6. **Ordering** — what to do first, second, third (dependency-aware)
7. **Expected outcomes** — what metrics should look like after each change

### What We DON'T Want
- Generic "try these hyperparameters" advice
- High-level architecture diagrams without implementation details
- Suggestions that require hardware we don't have
- Changes that only address symptoms without fixing root causes
- Incremental patches that don't address the fundamental learning problem

---

## KEY QUESTIONS TO ANSWER

1. **What is the SINGLE most impactful change we should make first?**

2. **Given our hardware constraints, what is the optimal architecture?**
   - Should we use a smaller backbone to free VRAM for better temporal modeling?
   - Should we drop VideoMAE and use a lighter temporal stream?
   - Should we reduce detection anchors to save compute?

3. **How do we prevent catastrophic multi-task interference?**
   - Is the shared backbone the problem? (Gradients from one head destroying another's features)
   - Should we use gradient surgery, PCGrad, or similar techniques?
   - Should some heads be more isolated (separate feature paths)?

4. **What is the minimum experiment that proves learning?**
   - Can we overfit a single batch on ALL heads simultaneously?
   - What does a "healthy" training curve look like for each head?
   - What are the early warning signs of collapse?

5. **Should we train from scratch or resume from the existing checkpoint?**
   - The backbone has useful ImageNet features
   - But the EMA-contaminated checkpoint may have corrupted backbone features
   - Is there value in the 43 epochs of training already done?

6. **How do we make the paper's thesis defensible?**
   - "One model matches N specialists" — what evidence do we need?
   - Which ablation rows are most important?
   - What is our strongest potential contribution?

---

## CONTEXT: THE PAPER WE'RE TARGETING

The paper (`popw_paper_improved.tex`) claims:
1. First unified architecture for multi-task assembly understanding
2. Two-stage FiLM conditioning for cross-task information flow
3. Kendall homoscedastic uncertainty weighting for stable joint optimization
4. Competitive accuracy vs dedicated single-task baselines

**Baselines to beat on IndustReal**:
- Detection: YOLOv8m → 83.80% mAP (b-boxed)
- Activity: MViTv2 → 65.25% Top-1 (RGB-only)
- PSR: B2 heuristic → F1=0.731, POS=0.816
- Head Pose: No baseline (free win)

**Our realistic targets**:
- Detection: ≥ 70% mAP (competitive, not necessarily beating)
- Activity: ≥ 45% Top-1 (with clip-level protocol)
- PSR: ≥ 0.60 F1 (beatable — B2 is a heuristic)
- Head Pose: ≤ 25° angular MAE (uncontested)

---

## FINAL INSTRUCTIONS

1. Read ALL provided files thoroughly before writing anything
2. Cross-reference the code with the root causes to verify the analysis
3. Produce at least 5 implementation guides as described above
4. Each guide should be 500+ lines of detailed, actionable content
5. Include actual code snippets that can be directly implemented
6. Reference specific line numbers in our codebase
7. Provide a clear execution order across all guides
8. End with a "Day 1 Action Plan" — what to do in the first 24 hours

**We are counting on you to break through the wall we've hit. Give us a path to a model that learns.**
